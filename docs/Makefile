# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for convenience
clean:
	rm -rf $(BUILDDIR)/*

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Generate API documentation
api:
	sphinx-apidoc -o $(SOURCEDIR)/api ../pado

# Build all documentation
all: clean api html

# Deploy to GitHub Pages
deploy:
	@echo "Deploying to GitHub Pages..."
	@if [ -d "$(BUILDDIR)/html" ]; then \
		git checkout gh-pages || git checkout -b gh-pages; \
		git rm -rf .; \
		cp -r $(BUILDDIR)/html/* .; \
		git add .; \
		git commit -m "Update documentation"; \
		git push origin gh-pages; \
		git checkout main; \
	else \
		echo "Documentation not built. Run 'make html' first."; \
	fi 