import logfire
from app.config import get_settings

s = get_settings()
t = s.LOGFIRE_TOKEN
print("token prefix:", repr(t[:12]), "| len:", len(t), "| whitespace:", t != t.strip())
print("send:", s.LOGFIRE_SEND_TO_LOGFIRE, "| service:", s.LOGFIRE_SERVICE_NAME, "| env:", s.LOGFIRE_ENVIRONMENT)

logfire.configure(token=t.strip(), send_to_logfire=True, service_name="smoke-test")
logfire.info("hello from saarthi smoke test")
logfire.force_flush()
print("done")
