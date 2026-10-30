//go:build linux

package main

import (
	"os/exec"
	"syscall"
)

// setChildDeath: вспомогательный процесс (nvidia-smi) умирает вместе с прокси.
func setChildDeath(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
}
