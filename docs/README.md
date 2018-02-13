# Writing documentation for Tensor Comprehensions

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
