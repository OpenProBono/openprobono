"""Set up logging formatter and functions for returns git commit hash and tags."""
import logging
import subprocess


class GitInfoFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', git_tag=None, git_hash=None):
        super().__init__(fmt, datefmt, style)
        self.git_tag = git_tag or get_git_tag()
        self.git_hash = git_hash or get_git_hash()


    def format(self, record):
        # Add git info to the log record
        record.git_tag = self.git_tag or "no-tag"
        record.git_hash = self.git_hash or "no-commit"
        return super().format(record)

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()
    except subprocess.CalledProcessError:
        return "unknown"

def get_git_tag():
    try:
        return subprocess.check_output(['git', 'describe', '--tags']).strip().decode()
    except subprocess.CalledProcessError:
        return None

def setup_logger():
    # Create a logger
    logger = logging.getLogger("logger")
    if not len(logger.handlers):
        logger.setLevel(logging.DEBUG)

        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create a formatter that includes git info
        formatter = GitInfoFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s [commit: %(git_hash)s, tag: %(git_tag)s]',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Set the formatter for the console handler
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

    return logger









def get_git_info():
    def get_git_revision_short_hash():
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()

    def get_git_tag():
        try:
            return subprocess.check_output(['git', 'describe', '--tags']).strip().decode()
        except subprocess.CalledProcessError:
            return None

    return {
        "commit_hash": get_git_revision_short_hash(),
        "tag": get_git_tag()
    }

