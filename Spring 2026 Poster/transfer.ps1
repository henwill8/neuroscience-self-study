wsl bash -lc "rsync -avz --delete \
  --exclude='.vscode' \
  --exclude='.git' \
  --exclude='node_modules' \
  --exclude='neural_ensembles' \
  --exclude='.venv' \
  --exclude='transfer.ps1' \
  -e 'ssh -J d00508545@ssh.cs.utahtech.edu' \
  ./ \
  'henry@144.38.192.52:~/projects/spring26/'"