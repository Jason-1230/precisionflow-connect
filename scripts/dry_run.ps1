param(
    [string]$Config = ".\configs\multinode_2x4.json",
    [int]$TensorElements = 1000000
)

python -m precisionflow_lab connect --manifest $Config --anonymize-hostnames
python -m precisionflow_lab inspect $Config --tensor-elements $TensorElements
