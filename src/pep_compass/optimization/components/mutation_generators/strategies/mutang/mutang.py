"""Fixed internal MUTANG action sequence."""


class Mutang:
    """Compose geometry, direction, scoring, selection, and combination stages."""

    def __init__(self, geometry, directions, scoring, selection, combination, autoencoder):
        self.geometry, self.directions = geometry, directions
        self.scoring, self.selection = scoring, selection
        self.combination, self.autoencoder = combination, autoencoder

    def generate(self, batch, context):
        geometries = self.geometry.resolve(batch, self.autoencoder)
        results = []
        for index, sequence in enumerate(batch.sequences):
            geometry = geometries[index]
            selected = self.directions.select(geometry.singular_values[0])
            scores = self.scoring.score(geometry.left_vectors[0], selected)
            options = self.selection.select(scores, len(sequence))
            generated = tuple(dict.fromkeys(self.combination.generate(sequence, options, context.rng)))
            results.append((generated, options))
        return results
