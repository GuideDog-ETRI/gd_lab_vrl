# 체크포인트

`teachers/<방법>/<묶음>/`, `students/<방법>/<묶음>/` 아래 README, `params/{agent,env}.yaml`, `SHA256SUMS.txt`,
`metadata.json`, leaderboard만 git에 있다. 가중치(`*.pt`, `*.onnx`)는 git 밖이다(.gitignore).
받은 가중치는 같은 경로에 두고 `sha256sum -c SHA256SUMS.txt`로 확인한다.
