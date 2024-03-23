import logging


def config_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )
    logging.info("Logger configured successfully")
