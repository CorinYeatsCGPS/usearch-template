import csv
import gzip
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from numba import carray, cfunc, njit, types
from usearch.compiled import MetricKind, MetricSignature, ScalarKind
from usearch.index import CompiledMetric, Index, Matches

@cfunc(
    types.float32(
        types.CPointer(types.float32), types.CPointer(types.float32), types.uint64
    )
)
@njit
def distance_score(a_ptr, b_ptr, ndim):
    # Read the pointers to the vectors into arrays to work with them.
    a_array = carray(a_ptr, ndim)
    b_array = carray(b_ptr, ndim)
    # Fill in the rest here and return the distance score.
    pass


def build(profile_file: Path, profile_size: int, index_file: Path, index_batch_size = 10000):

    # Create an index instance with the given number of dimensions and the distance score function.
    index = Index(
        ndim=profile_size,
        dtype=ScalarKind.F32,  # data type for internal vector storage
        metric=CompiledMetric(
            pointer=distance_score.address,
            kind=MetricKind.Unknown,  # Indicates a bespoke metric in this case
            signature=MetricSignature.ArrayArraySize,
            # Can either be ArrayArraySize or ArrayArray, depending on whether you need to pass in the number of
            # dimensions. If using ArrayArray, change the distance function signature correspondingly
        ),
    )

    count = 0
    for keys, profiles in batch_read_profiles(
        profile_file, batch_size=index_batch_size # Indexes are added in batches as a trade-off between memory usage and indexing speed
    ):
        index.add(keys, profiles)
        count += index_batch_size
        print(f"Indexed {count:,} profiles", file=sys.stderr)

    print("Saving index...", file=sys.stderr)
    index.save(index_file)
    print(
        f"Successfully indexed {count:,} profiles and saved to {index_file.name}",
        file=sys.stderr,
    )


def batch_read_profiles(profiles_file: Path, batch_size: int = 10000) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Parses a gzipped tab-separated profiles file into numpy arrays keyed by the ST.
       The profiles are read in batches for efficiency when inserting them into the index"""

    with gzip.open(profiles_file, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip header

        batch_keys = []
        batch_vectors = []
        for row, line in enumerate(reader):
            key = int(line[0]) # ST code
            vector = np.array(list(map(float, line[1:])), dtype=np.float32) # Vector of profile
            batch_keys.append(key)
            batch_vectors.append(vector)

            if len(batch_keys) == batch_size:
                yield np.array(batch_keys), np.array(batch_vectors)
                batch_keys = []
                batch_vectors = []

        # Yield any remaining profiles
        if batch_keys:
            yield np.array(batch_keys), np.array(batch_vectors)


def search(index_file: Path, input_profile: list[int], num_matches: int):
    index = Index.restore(index_file, view=True)
    index.metric = distance_score
    query_vector = np.array(input_profile, dtype=np.float32)
    matches: Matches = index.search(query_vector, num_matches)
    for match in matches:
        print(f"Match: ST {match.key}, Distance: {match.distance}: {str(match)}")
