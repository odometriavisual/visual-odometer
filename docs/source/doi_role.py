# docs/doi_role.py

from docutils import nodes
from docutils.parsers.rst import roles

try:
    # Use SciPy's internal DOI resolver if available
    from scipy._lib.doi_roi import doi_url
except ImportError:
    def doi_url(doi: str) -> str:
        """Fallback DOI resolver."""
        return f"https://doi.org/{doi.strip()}"

def doi_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Custom role for linking DOIs in Sphinx docs."""
    url = doi_url(text.strip())
    node = nodes.reference(rawtext, f"DOI:{text}", refuri=url, **options)
    return [node], []

def setup(app):
    """Register the :doi: role."""
    roles.register_local_role("doi", doi_role)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
