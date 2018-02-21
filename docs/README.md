# Generating Sphinx documentation for Tensor Comprehensions

1. First install sphinx

```Shell
pip install Sphinx
```

2. We use mobile friendly sphinx_rtd_theme style theme available here https://github.com/rtfd/sphinx_rtd_theme. Run the command below:

```Shell
cd docs && pip install -r requirements.txt
```

3. Edit the document under `docs/source/`. If you are adding a new doc, make sure to modify toctree in `source/index.rst`. For syntax about `.rst`, this link might be helpful http://www.sphinx-doc.org/en/stable/rest.html. If you have written markdown before, it should be easy to write `.rst` files as well.

**TIP**: All the sphinx based documentations have a link `View Page Source` on top right. You can click that link to see the corresponding `.rst` file for that page.

4. Run
```Shell
cd docs && make html
```

5. Now you can see the generated html `index.html` under `build/html/`

6. Send PR

# Generating Doxygen docs for Tensor Comprehensions

1. Install Doxygen

Run the command

```Shell
$ apt-get install doxygen
```

2. Edit the `docs/doxygen/index.md` file for making changes to the main page for
doxygen docs.

3. Edit the `docs/doxygen/Doxyfile` file for making change to what code should be
documented, excluded etc.

4. Now, test the docs locally. Run the following commands:

```Shell
$ cd $HOME/TensorComprehensions && mkdir -p $HOME/TensorComprehensions-docs/api
$ doxygen docs/doxygen/Doxyfile
```

This will generate an `html` folder which will contain all the html files for the
documentation. Please `DO NOT` edit the `html` folder manually. Rather make changes
as suggested in Step 2, 3 and re-generate docs.

5. Check the HTML docs look fine to you.

6. Send a PR
