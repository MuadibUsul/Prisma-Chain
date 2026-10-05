"""prisma: one CLI entrypoint over the Prisma product packages."""

from .version_cmd import cmd_version

__all__ = ["cmd_version"]
