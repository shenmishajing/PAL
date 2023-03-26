from mmengine.hub import get_config

if __name__ == "__main__":
    get_config("", True).dump("config.yaml")
