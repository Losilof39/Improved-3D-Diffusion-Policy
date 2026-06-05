try:
    from huggingface_hub import cached_download  # noqa: F401
except ImportError:
    # huggingface_hub>=0.23.0 removed cached_download; patch it back so that
    # diffusers<=0.20.x (which still references it) can import cleanly.
    import huggingface_hub
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download
