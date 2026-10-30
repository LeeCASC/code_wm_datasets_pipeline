import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
import world_model_pipeline.tools.download as direct


class DirectDownloadTests(unittest.TestCase):
    def setUp(self):
        self.payload=bytes(range(256))*32
        self.seen=[]
        outer=self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                start,end=map(int,self.headers['Range'].split('=')[1].split('-'))
                outer.seen.append(start)
                self.send_response(206)
                self.send_header('Content-Range',f'bytes {start}-{end}/{len(outer.payload)}')
                self.send_header('Content-Length',str(end-start+1));self.end_headers()
                self.wfile.write(outer.payload[start:end+1])
            def log_message(self,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}/public.bin'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def test_resume_and_ignore_proxy_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'model.bin';partial=target.with_suffix('.bin.part')
            partial.write_bytes(self.payload[:1024]+b'\0'*(len(self.payload)-1024))
            expected=hashlib.sha256(self.payload).hexdigest()
            state=dict(url=self.url,size=len(self.payload),sha256=expected,chunk_size=1024,completed=[0])
            target.with_suffix('.bin.parts.json').write_text(json.dumps(state))
            with patch.dict('os.environ',{'HTTP_PROXY':'http://127.0.0.1:9','HTTPS_PROXY':'http://127.0.0.1:9','ALL_PROXY':'http://127.0.0.1:9','NO_PROXY':''}):
                direct.download(self.url,target,len(self.payload),expected,workers=3,chunk_size=1024)
            self.assertEqual(target.read_bytes(),self.payload)
            self.assertNotIn(0,self.seen)
            report=json.loads(target.with_suffix('.download.json').read_text())
            self.assertTrue(report['proxies_disabled'])
            self.assertFalse(report['trust_env'])

    def test_bad_hash_is_never_promoted_to_final_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'model.bin'
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                direct.download(self.url,target,len(self.payload),'0'*64,workers=2,chunk_size=2048)
            self.assertFalse(target.exists())
            self.assertTrue(target.with_suffix('.bin.part').exists())


if __name__=='__main__':unittest.main()
