import logging
import traceback

APPLICATION_LOG_PREFIX = " FILMBASE_API_LOG: "


class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        exception_str = f": {traceback.format_exc()}"

        return (
            APPLICATION_LOG_PREFIX
            + " "
            + str(self.logger.findCaller())
            + " "
            + str(msg)
            + " "
            + exception_str,
            kwargs,
        )


def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    adapter = ContextAdapter(logger, {})

    return adapter
