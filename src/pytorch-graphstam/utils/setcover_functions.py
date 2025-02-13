# Helper functions that operate on sets & lists and aid batched training/testing to save GPU Memory

def greedy_set_cover(list_of_lists):
    """
    Returns a subset of `list_of_lists` that covers all elements present
    in the union of `list_of_lists`, using a greedy strategy.

    :param list_of_lists: A list of lists (or sets) of elements.
    :return: A list of the chosen sub-lists whose union covers the entire set of elements.
    """

    # Convert all lists to sets for faster set operations
    sets = [set(lst) for lst in list_of_lists]

    # Compute the union (all elements that need to be covered)
    universe = set().union(*sets)

    chosen_sets = []  # This will store the sub-lists we pick

    # Work with a local copy of 'sets' if you don't want to modify the original
    remaining_sets = sets[:]

    # While there are still elements to cover in the universe
    while universe:
        # Pick the set that covers the largest number of still-uncovered elements
        best_set = max(remaining_sets, key=lambda s: len(s & universe))

        # Add this set to the chosen list
        chosen_sets.append(best_set)

        # Remove the covered elements from the universe
        universe -= best_set

        # Remove that set from further consideration
        remaining_sets.remove(best_set)

        # If there are no remaining sets but the universe is not empty,
        # it means we cannot cover all elements (this can happen if there's an element
        # that doesn't appear in any set). Check for that if needed.

    # Optionally, you can convert chosen_sets back to lists if you prefer
    chosen_sets = [list(s) for s in chosen_sets]

    return chosen_sets

def merge_subsets_min_bins(subsets, x):
    """
    Given an iterable of subsets, merge them into as few bins as possible
    so that the union of subsets in each bin has at most 'x' elements.
    Merging is allowed whether or not the subsets overlap.

    :param subsets: Iterable of sets (or frozensets) of elements
    :param x: Maximum allowed size (cardinality) of the union in each bin
    :return: A list of sets, each representing the union of subsets assigned to that bin
    """
    # 1. Sort the subsets in descending order by size (largest first)
    #    Converting each to frozenset so it's immutable (optional, for safety)
    sorted_subsets = sorted((frozenset(s) for s in subsets), key=lambda s: len(s), reverse=True)

    # 2. List of "bins", where each bin is a set that represents the union
    #    of subsets assigned to it
    bins = []

    # 3. Greedily place each subset into the first bin that can accommodate it
    for subset in sorted_subsets:
        placed = False
        for i, b in enumerate(bins):
            # new union size if we add 'subset' to bin 'b'
            new_union_size = len(b.union(subset))
            if new_union_size <= x:
                # Merge into this bin
                bins[i] = b.union(subset)
                placed = True
                break
        if not placed:
            # Create a new bin with this subset
            bins.append(subset)

    # Convert each bin's union back to a normal set (if desired)
    return [set(b) for b in bins]


def list_of_lists_to_set_of_subsets(list_of_lists):
    """
    Convert a list of lists into a set of subsets (using frozenset to be hashable).
    """
    return {frozenset(lst) for lst in list_of_lists}

def set_of_subsets_to_list_of_lists(set_of_subsets):
    """
    Convert a set of frozensets (or sets) into a list of lists.
    """
    # Convert each subset (frozenset) to a list, then collect all in a top-level list
    return [list(subset) for subset in set_of_subsets]

def list_of_sets_to_list_of_lists(list_of_sets):
    """
    Convert a list of sets into a list of lists.

    Note: The order of elements in each resulting list is not guaranteed,
          because Python's built-in 'set' does not maintain a guaranteed order.
    """
    return [list(s) for s in list_of_sets]
