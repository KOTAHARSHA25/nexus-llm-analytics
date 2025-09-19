# Shared utility functions for backend
import datetime
import os
import json
import logging

class JsonFormatter(logging.Formatter):
	def format(self, record):
		log_record = {
			'level': record.levelname,
			'time': self.formatTime(record, self.datefmt),
			'name': record.name,
			'message': record.getMessage(),
		}
		if record.exc_info:
			log_record['exception'] = self.formatException(record.exc_info)
		return json.dumps(log_record)

# Data versioning/audit trail
def log_data_version(event, filename, details=None):
	audit_dir = os.path.join(os.path.dirname(__file__), '../../data/audit')
	os.makedirs(audit_dir, exist_ok=True)
	entry = {
		'event': event,
		'filename': filename,
		'details': details,
		'timestamp': datetime.datetime.now().isoformat()
	}
	with open(os.path.join(audit_dir, 'audit_log.jsonl'), 'a') as f:
		f.write(json.dumps(entry) + '\n')

# Configurable logging level
def setup_logging(logfile='nexus.log'):
	level = os.environ.get('LOG_LEVEL', 'INFO').upper()
	logger = logging.getLogger()
	logger.setLevel(level)
	# Console handler
	ch = logging.StreamHandler()
	ch.setLevel(level)
	ch.setFormatter(JsonFormatter())
	# File handler
	fh = logging.FileHandler(logfile)
	fh.setLevel(level)
	fh.setFormatter(JsonFormatter())
	# Avoid duplicate handlers
	logger.handlers.clear()
	logger.addHandler(ch)
	logger.addHandler(fh)

# User-friendly error messages
def friendly_error(msg, suggestion=None):
	return {'error': msg, 'suggestion': suggestion or 'Please check your input and try again.'}

# Pluggable agent registry
class AgentRegistry:
	def __init__(self):
		self.registry = {}
	def register(self, name, agent):
		self.registry[name] = agent
	def get(self, name):
		return self.registry.get(name)


