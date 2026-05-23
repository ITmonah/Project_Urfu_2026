def process_images(images):
    from .pipeline import process_images as _process_images

    return _process_images(images)


__all__ = ["process_images"]
