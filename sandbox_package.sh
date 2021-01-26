rm work.tar.gz
tar --exclude __pycache__ --exclude=./logs --exclude=**.log --exclude=./venv --exclude=./.gitignore --exclude=./.git --exclude=./.idea --exclude=sandbox_update.sh --exclude=sandbox_package.sh --exclude=work.tar.gz -zcvf work.tar.gz .
tar -tf work.tar.gz