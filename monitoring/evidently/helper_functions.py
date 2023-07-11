def get_evidently_html(evidently_object) -> str:
    """Returns the rendered EvidentlyAI report/metric as HTML

    Should be assigned to `self.html`, installing `metaflow-card-html` to be rendered
    """
    import tempfile

    with tempfile.NamedTemporaryFile() as tmp:
        evidently_object.save_html(tmp.name)
        with open(tmp.name) as fh:
            return fh.read()
