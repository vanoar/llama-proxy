//go:build !linux

package main

import "os/exec"

func setChildDeath(cmd *exec.Cmd) {}
