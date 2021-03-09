#! /bin/bash
kill -9 `ps auxw|grep ray|grep -v grep|awk '{print $2}'`
