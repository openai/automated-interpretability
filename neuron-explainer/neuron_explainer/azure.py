def standardize_azure_url(url):
    """Make sure url is converted to url format, not an azure path"""
    if url.startswith("az://openaipublic/"):
        url = url.replace("az://openaipublic/", "https://openaipublic.blob.core.windows.net/")
    return url
