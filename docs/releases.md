# Releasing a package update

Make changes to the code taking care to add [tests](https://gitlabsbnb.irbbarcelona.org/alenes/signaturizer3d/-/tree/main/tests) if you are adding new functionality 
or fixing a bug.

Ensure the test pass, run them locally with:
```shell
poetry run pytest
# you can also tests that all files are formatted correctly by running black
poetry run black .
```

Commit your changes and push them to gitlab.
On push CI checks are run that run the tests and check that the code is formatted with Black, make sure these check pass.

Once you're happy with your changes you can bump the package version with poetry. The package (tries) to use [semantic versioning](https://semver.org/),
this means the types of changes you've made determine what part of the package version number you'll update. For a small backwards compatible
fix you can bump the patch version, for adding a feature in a backwards compatible way bump the minor version, for changes that break
backwards compatibility bump the major version. Bump the version in poetry like this:
```shell
poetry version patch
```
Check in and commit the updated version number (the file that changes is `pyproject.toml`).
Push your changes to gitlab, then create a release tag for your new release number and push that to gitlab.

```shell
# Assuming your new version number is 0.1.3, you would run
git tag 0.1.3
git push origin 0.1.3
```
There is a job set up in gitlab that publishes the package to pypi test automatically each time a new tag is pushed.