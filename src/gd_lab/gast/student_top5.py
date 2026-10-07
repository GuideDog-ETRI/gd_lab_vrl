"""Top-5 student checkpoints ranked by smoothed training objective."""
from collections import deque
import json, math, os
from pathlib import Path

class StudentTop5:
    def __init__(self, directory, *, start_iteration=5000, keep=5, smoothing_windows=8,
                 min_visible_fraction=.95, min_visible_sample_fraction=None,
                 min_hazard_supervised_fraction=.95, score_description=None, min_score_rows=0):
        if start_iteration < 0 or keep < 1 or smoothing_windows < 1:
            raise ValueError("invalid Top-5 settings")
        if ((min_visible_fraction is not None and not 0 <= min_visible_fraction <= 1)
                or (min_visible_sample_fraction is not None and not 0 <= min_visible_sample_fraction <= 1)
                or not 0 <= min_hazard_supervised_fraction <= 1):
            raise ValueError("quality thresholds must be in [0, 1]")
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.start_iteration=start_iteration; self.keep=keep
        self.history=deque(maxlen=smoothing_windows)
        self.min_visible_fraction=min_visible_fraction
        self.min_visible_sample_fraction=min_visible_sample_fraction
        self.min_hazard_supervised_fraction=min_hazard_supervised_fraction
        self.criteria={"version":2,"start_iteration":start_iteration,"keep":keep,
            "rolling_optimizer_windows":smoothing_windows,
            "score":score_description or "latent_mse + hazard_loss_coef * hazard_mse + extra_loss",
            "order":"ascending (lower is better)",
            "min_visible_fraction":min_visible_fraction,
            "min_visible_sample_fraction":min_visible_sample_fraction,
            "visibility_gate_semantics":(
                "fraction of student rows with at least one teacher-visible height cell"
                if min_visible_sample_fraction is not None
                else "mean fraction of map cells marked visible (legacy gate)"),
            "min_hazard_supervised_fraction":min_hazard_supervised_fraction,
            "score_scope":"training windows, not held-out validation"}
        self.min_score_rows=int(min_score_rows)
        if self.min_score_rows:
            self.criteria["min_score_rows_per_window"]=self.min_score_rows
        self._write(self.directory/"criteria.json",self.criteria)
        self.entries=self._load()

    @staticmethod
    def _write(path,value):
        tmp=path.with_name(path.name+".tmp")
        tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
        os.replace(tmp,path)

    def _load(self):
        try: entries=json.loads((self.directory/"leaderboard.json").read_text()).get("entries",[])
        except (OSError,ValueError,TypeError): return []
        valid=[]
        for e in entries:
            try: score=float(e["score"]); name=str(e["file"])
            except (KeyError,TypeError,ValueError): continue
            if math.isfinite(score) and (self.directory/name).is_file():
                valid.append({**e,"score":score})
        return sorted(valid,key=lambda e:(e["score"],e["iteration"]))[:self.keep]

    def _persist(self):
        self.entries.sort(key=lambda e:(e["score"],e["iteration"]))
        self.entries=self.entries[:self.keep]
        self._write(self.directory/"leaderboard.json",{"criteria":self.criteria,"entries":self.entries})
        retained={e["file"] for e in self.entries}
        for path in self.directory.glob("student_top5_iter_*.pt"):
            if path.name not in retained: path.unlink(missing_ok=True)

    def consider(self,iteration,score,metrics,save_fn):
        if iteration < self.start_iteration:
            return {"evaluated":False,"saved":False,"reason":"before_start"}
        try:
            score=float(score); updates=int(metrics["updates"])
            vals={k:float(metrics[k]) for k in ("latent_mse","hazard_mse","extra_loss",
                "visible_fraction","hazard_supervised_fraction")}
            if "visible_sample_fraction" in metrics:
                vals["visible_sample_fraction"] = float(metrics["visible_sample_fraction"])
        except (KeyError,TypeError,ValueError):
            return {"evaluated":True,"saved":False,"reason":"invalid_metrics"}
        if updates<=0 or not all(math.isfinite(v) for v in [score,*vals.values()]):
            return {"evaluated":True,"saved":False,"reason":"invalid_metrics"}
        if self.min_score_rows and int(metrics.get("score_rows",0))<self.min_score_rows:
            return {"evaluated":True,"saved":False,"reason":"too_few_score_rows"}
        if self.min_visible_sample_fraction is not None:
            if "visible_sample_fraction" not in vals:
                return {"evaluated":True,"saved":False,"reason":"missing_visibility_sample_coverage"}
            if vals["visible_sample_fraction"] < self.min_visible_sample_fraction:
                return {"evaluated":True,"saved":False,"reason":"low_visibility_sample_coverage"}
        elif self.min_visible_fraction is not None and vals["visible_fraction"] < self.min_visible_fraction:
            return {"evaluated":True,"saved":False,"reason":"low_visibility"}
        if vals["hazard_supervised_fraction"]<self.min_hazard_supervised_fraction:
            return {"evaluated":True,"saved":False,"reason":"low_hazard_supervision"}
        self.history.append({"score":score,**vals})
        avg={k:sum(row[k] for row in self.history)/len(self.history)
             for k in self.history[0]}
        avg["windows"]=len(self.history)
        if len(self.history)<self.history.maxlen:
            return {"evaluated":True,"saved":False,"reason":"smoothing_warmup",**avg}
        if any(int(e["iteration"])==iteration for e in self.entries):
            return {"evaluated":True,"saved":False,"reason":"iteration_already_ranked",**avg}
        if len(self.entries)>=self.keep and avg["score"]>=self.entries[-1]["score"]:
            return {"evaluated":True,"saved":False,"reason":"outside_top5",**avg}
        name=f"student_top5_iter_{iteration:05d}.pt"
        record={"iteration":iteration,"score":avg["score"],"metrics":avg,"file":name}
        save_fn(self.directory/name,record)
        self.entries.append(record); self._persist()
        rank=next(i for i,e in enumerate(self.entries,1) if e["file"]==name)
        return {"evaluated":True,"saved":True,"rank":rank,**avg}
