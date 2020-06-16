import nussl
import gin

make_configurable = []

for x in dir(nussl.ml.networks.builders):
    if x.startswith('build'):
        make_configurable.append({
            'bind': getattr(nussl.ml.networks.builders, x),
            'module': 'nussl.ml.networks.builders'
        })

exclude_from_datasets = ['base_dataset', 'hooks', 'transforms']

for x in dir(nussl.datasets):
    if x not in exclude_from_datasets and not x.startswith('_'):
        make_configurable.append({
            'bind': getattr(nussl.datasets, x),
            'module': 'nussl.datasets'
        })

exclude_from_transforms = ['compute_ideal_binary_mask', 'logging',
 'np', 'numcodecs', 'os', 'random', 'shutil', 'time_frequency_keys', 
 'torch', 'utils', 'zarr', 'OrderedDict', 'TransformException']

for x in dir(nussl.datasets.transforms):
    if x not in exclude_from_transforms and not x.startswith('_'):
        make_configurable.append({
            'bind': getattr(nussl.datasets.transforms, x),
            'module': 'nussl.datasets.transforms'
        })

for val in make_configurable:
    gin.external_configurable(
        val['bind'], module=val['module'])

# Primitives
module = 'nussl.separation.primitive'
gin.external_configurable(nussl.separation.primitive.FT2D, module=module)
gin.external_configurable(nussl.separation.primitive.Repet, module=module)
gin.external_configurable(nussl.separation.primitive.RepetSim, module=module)
gin.external_configurable(nussl.separation.primitive.TimbreClustering, module=module)
gin.external_configurable(nussl.separation.primitive.Melodia, module=module)
gin.external_configurable(nussl.separation.primitive.HPSS, module=module)

# Spatial
module = 'nussl.separation.spatial'
gin.external_configurable(nussl.separation.spatial.SpatialClustering, module=module)
gin.external_configurable(nussl.separation.spatial.Projet, module=module)
gin.external_configurable(nussl.separation.spatial.Duet, module=module)

# Benchmark
module = 'nussl.separation.benchmark'
gin.external_configurable(nussl.separation.benchmark.HighLowPassFilter, module=module)
gin.external_configurable(nussl.separation.benchmark.IdealBinaryMask, module=module)
gin.external_configurable(nussl.separation.benchmark.IdealRatioMask, module=module)
gin.external_configurable(nussl.separation.benchmark.WienerFilter, module=module)
gin.external_configurable(nussl.separation.benchmark.MixAsEstimate, module=module)

# Deep
module = 'nussl.separation.deep'
gin.external_configurable(nussl.separation.deep.DeepClustering, module=module)
gin.external_configurable(nussl.separation.deep.DeepMaskEstimation, module=module)
gin.external_configurable(nussl.separation.deep.DeepAudioEstimation, module=module)

# Composite
module = 'nussl.separation.composite'
gin.external_configurable(nussl.separation.composite.EnsembleClustering, module=module)

# Factorization
module = 'nussl.separation.factorization'
gin.external_configurable(nussl.separation.factorization.RPCA, module=module)
gin.external_configurable(nussl.separation.factorization.ICA, module=module)

# Evaluation
module = 'nussl.evaluation'
gin.external_configurable(nussl.evaluation.BSSEvalScale, module=module)
gin.external_configurable(nussl.evaluation.BSSEvalV4, module=module)
gin.external_configurable(nussl.evaluation.PrecisionRecallFScore, module=module)

# Closures
module = 'nussl.ml.train.closures'
gin.external_configurable(nussl.ml.train.closures.TrainClosure, module=module)
gin.external_configurable(nussl.ml.train.closures.ValidationClosure, module=module)
gin.external_configurable(nussl.ml.train.closures.Closure, module=module)

# General nussl things
gin.external_configurable(nussl.AudioSignal, module='nussl')
gin.external_configurable(nussl.STFTParams, module='nussl')
gin.external_configurable(nussl.ml.SeparationModel, module='nussl.ml')
