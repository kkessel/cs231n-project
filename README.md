# cs231n-project

This repo contains submodules. If you're cloning it for the first time, use 

`git clone --recursive`

If you have already cloned it, the submodules were recently added, so run 

`git submodule update --init`

If the submodule were updated, which should not happen in our case, you would
need to do something like one of the following:

```
# pull all changes in the repo including changes in the submodules
git pull --recurse-submodules

# pull all changes for the submodules
git submodule update --remote
```
