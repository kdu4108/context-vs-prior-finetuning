from typing import Callable, List, Optional, Tuple


def format_query(query: str, entity: Tuple[str], context: str, prefix="", answer=None):
    """
    Number of elements in entity must match the number of format {} things in query.
    This is to handle for multiple-entity entities (e.g. friend enemy pairs)
    """
    if not isinstance(entity, tuple):
        raise ValueError("entity must be of type tuple.")
    if "{entity}" in query:
        if "{answer}" in query:
            if answer is None:
                raise ValueError("Expected answer to be provided because query contains {answer} but none was given.")
            concrete_query = query.format(entity=entity[0], answer=answer)
        else:
            concrete_query = query.format(entity=entity[0])
    else:
        if answer is not None:
            concrete_query = query.format(*entity, answer=answer)
        else:
            concrete_query = query.format(*entity)
    return prefix + context + concrete_query
