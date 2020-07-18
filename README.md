Add the following to `.git/hooks/pre-push`

```
kaggle kernels push
```

and remember not to push unless you intend to run a new version of a notebook.

If you want to push WITHOUT running the notebook, you can do `git push origin master --no-verify`.
