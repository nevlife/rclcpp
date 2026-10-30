# Copyright 2026 nev
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check hybrid delivery with a remote publisher and enabled and disabled readers."""

import array
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import _ipc_native
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

TOPIC = '/ipc_hybrid'
SAMPLES = 20
EXPECTED = Counter({('cpp', i): 1 for i in range(SAMPLES)})
EXPECTED.update({('remote', 1000 + i): 1 for i in range(SAMPLES)})


def key(message):
    """Validate the complete payload and return its unique source/sample identity."""
    assert len(message.data) == 1024 and message.data.count(42) == 1024
    return message.header.frame_id, message.height


def child(directory):
    """Run a second-process RMW publisher and reader against the same topic."""
    rclpy.init()
    node = rclpy.create_node('remote_component', start_parameter_services=False,
                             enable_rosout=False)
    received = Counter()
    node.create_subscription(Image, TOPIC, lambda msg: received.update([key(msg)]), 100)
    pub = node.create_publisher(Image, TOPIC, 100)
    end = time.monotonic() + 10
    while pub.get_subscription_count() < 3:
        assert time.monotonic() < end, pub.get_subscription_count()
        time.sleep(.01)
    (directory / 'ready').write_text('ready')
    while not (directory / 'start').exists():
        assert time.monotonic() < end
        time.sleep(.01)
    for i in range(SAMPLES):
        message = Image()
        message.data = array.array('B', [42]) * 1024
        message.header.frame_id = 'remote'
        message.height = 1000 + i
        pub.publish(message)
        rclpy.spin_once(node, timeout_sec=.005)
    while sum(received.values()) < 2 * SAMPLES:
        assert time.monotonic() < end, received
        rclpy.spin_once(node, timeout_sec=.05)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=.01)
    assert received == EXPECTED, received
    node.destroy_node()
    rclpy.shutdown()
    # zenoh may log to stdout before this line; the parent reads the last line.
    print(json.dumps({'remote_samples': sum(received.values())}))


def main():
    """Verify exactly-once callbacks, disabled-reader reachability and remote traffic."""
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        child(Path(sys.argv[2]))
        return
    native = _ipc_native.Container(True, False, False, TOPIC)
    assert native.enabled()
    context = Context()
    context.init(context_lease=native.lease(), initialize_logging=False)
    node = rclpy.create_node('local_python', context=context,
                             start_parameter_services=False, enable_rosout=False)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    typed = Counter()
    raw = Counter()
    node.create_subscription(Image, TOPIC, lambda msg: typed.update([key(msg)]), 100)
    node.create_subscription(
        Image, TOPIC, lambda data: raw.update([key(deserialize_message(data, Image))]),
        100, raw=True)
    with tempfile.TemporaryDirectory(prefix='wp4-remote-') as temporary:
        directory = Path(temporary)
        env = dict(os.environ, RCL_INTRA_PROCESS='0')
        process = subprocess.Popen([sys.executable, __file__, '--child', temporary],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        try:
            end = time.monotonic() + 15
            while not (directory / 'ready').exists():
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < end, 'remote discovery timeout'
                time.sleep(.01)
            (directory / 'start').write_text('start')
            for i in range(SAMPLES):
                native.publish(1024, i)
                executor.spin_once(timeout_sec=.005)
            while sum(typed.values()) < 2 * SAMPLES or sum(raw.values()) < 2 * SAMPLES:
                assert time.monotonic() < end, (typed, raw)
                executor.spin_once(timeout_sec=.05)
            for _ in range(20):
                executor.spin_once(timeout_sec=.01)
            output, errors = process.communicate(timeout=5)
            assert process.returncode == 0, (output, errors)
            assert typed == EXPECTED, typed
            assert raw == EXPECTED, raw
            print(json.dumps({'typed_samples': sum(typed.values()),
                              'disabled_reader_samples': sum(raw.values()),
                              'child': json.loads(output.splitlines()[-1])}))
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    executor.shutdown()
    node.destroy_node()
    context.destroy()
    native.close()


if __name__ == '__main__':
    main()
