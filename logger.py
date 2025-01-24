import neptune

# use whatever logger you prefer

run = neptune.init_run(
    project="",
    api_token="",
)  # your credentials

params = {"optimizer": "Adam"}
run["parameters"] = params