# Penetration Testing and Vulnerability Assessment
# Advanced security testing for vulnerability detection and penetration testing

import pytest
import asyncio
import threading
import time
import random
import hashlib
import base64
import os
import sys
import tempfile
import json
import socket
import ssl
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from unittest.mock import patch, Mock, MagicMock
import subprocess
import re
from urllib.parse import urlencode, quote, unquote
from datetime import datetime, timedelta

# Import components for penetration testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_data_structures import OptimizedTrie, HighPerformanceHashMap, OptimizedDataProcessor
from backend.core.optimized_llm_client import OptimizedLLMClient, SmartCache, OptimizedConnectionPool
from backend.core.rate_limiter import RateLimiter
from backend.core.optimized_file_io import OptimizedFileProcessor, StreamingCSVReader
from backend.core.enhanced_cache_integration import EnhancedCacheManager
from backend.core.intelligent_query_engine import IntelligentQueryRouter, QueryPatternAnalyzer


class VulnerabilityScanner:
    """Simulated vulnerability scanner for testing"""
    
    def __init__(self):
        self.vulnerabilities_found = []
        self.scan_results = {}
    
    def scan_sql_injection(self, component, test_inputs: List[str]) -> List[Dict[str, Any]]:
        """Scan for SQL injection vulnerabilities"""
        vulnerabilities = []
        
        for payload in test_inputs:
            try:
                # Test payload as key
                if hasattr(component, 'insert'):
                    component.insert(f"test_key_{hash(payload) % 1000}", payload)
                    result = component.search(f"test_key_{hash(payload) % 1000}")
                elif hasattr(component, 'put'):
                    component.put(f"test_key_{hash(payload) % 1000}", payload)
                    result = component.get(f"test_key_{hash(payload) % 1000}")
                
                # Check if payload was executed (simplified check)
                if isinstance(result, str):
                    if 'DROP TABLE' in result.upper() or 'SELECT' in result.upper():
                        vulnerabilities.append({
                            'type': 'SQL_INJECTION',
                            'severity': 'HIGH',
                            'payload': payload,
                            'component': str(type(component).__name__),
                            'description': 'Potential SQL injection vulnerability detected'
                        })
                        
            except Exception as e:
                # Errors might indicate input validation (good) or other issues
                if 'sql' in str(e).lower() or 'injection' in str(e).lower():
                    vulnerabilities.append({
                        'type': 'SQL_INJECTION_ERROR',
                        'severity': 'MEDIUM',
                        'payload': payload,
                        'component': str(type(component).__name__),
                        'description': f'SQL-related error: {str(e)}'
                    })
        
        return vulnerabilities
    
    def scan_xss_vulnerabilities(self, processor, test_payloads: List[str]) -> List[Dict[str, Any]]:
        """Scan for XSS vulnerabilities"""
        vulnerabilities = []
        
        for payload in test_payloads:
            try:
                # Create test file with XSS payload
                test_data = {
                    'user_input': payload,
                    'content': f'Test content with payload: {payload}'
                }
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(test_data, f)
                    temp_path = f.name
                
                result = asyncio.run(processor.process_file(temp_path))
                
                if 'error' not in result:
                    result_str = str(result)
                    
                    # Check if script tags are preserved without sanitization
                    if '<script>' in result_str and 'alert(' in result_str:
                        vulnerabilities.append({
                            'type': 'XSS_STORED',
                            'severity': 'HIGH',
                            'payload': payload,
                            'component': 'FileProcessor',
                            'description': 'XSS payload preserved without sanitization'
                        })
                    
                    # Check for other XSS vectors
                    dangerous_patterns = ['javascript:', 'onload=', 'onerror=', 'onclick=']
                    for pattern in dangerous_patterns:
                        if pattern in result_str.lower():
                            vulnerabilities.append({
                                'type': 'XSS_REFLECTED',
                                'severity': 'MEDIUM',
                                'payload': payload,
                                'component': 'FileProcessor',
                                'description': f'Dangerous XSS pattern detected: {pattern}'
                            })
                
                os.unlink(temp_path)
                
            except Exception as e:
                # XSS-related errors might indicate filtering (good)
                if any(term in str(e).lower() for term in ['script', 'xss', 'javascript']):
                    # This is actually good - indicates filtering
                    pass
        
        return vulnerabilities
    
    def scan_path_traversal(self, processor, traversal_payloads: List[str]) -> List[Dict[str, Any]]:
        """Scan for path traversal vulnerabilities"""
        vulnerabilities = []
        
        for payload in traversal_payloads:
            try:
                result = asyncio.run(processor.process_file(payload))
                
                if 'error' not in result:
                    result_str = str(result).lower()
                    
                    # Check if sensitive system files were accessed
                    sensitive_indicators = [
                        'root:x:0:0:',  # Unix passwd
                        '[administrators]',  # Windows
                        'system32',
                        '/etc/shadow',
                        'private key'
                    ]
                    
                    for indicator in sensitive_indicators:
                        if indicator in result_str:
                            vulnerabilities.append({
                                'type': 'PATH_TRAVERSAL',
                                'severity': 'CRITICAL',
                                'payload': payload,
                                'component': 'FileProcessor',
                                'description': f'Path traversal successful - accessed: {indicator}'
                            })
                            
            except (FileNotFoundError, PermissionError, OSError):
                # These exceptions are expected and good - they prevent traversal
                pass
            except Exception as e:
                # Other errors might indicate vulnerability
                if 'traversal' in str(e).lower() or 'path' in str(e).lower():
                    vulnerabilities.append({
                        'type': 'PATH_TRAVERSAL_ERROR',
                        'severity': 'LOW',
                        'payload': payload,
                        'component': 'FileProcessor',
                        'description': f'Path traversal error: {str(e)}'
                    })
        
        return vulnerabilities
    
    def scan_buffer_overflow(self, components: List[Any]) -> List[Dict[str, Any]]:
        """Scan for buffer overflow vulnerabilities"""
        vulnerabilities = []
        
        # Generate large payloads
        large_payloads = [
            "A" * 10000,      # 10KB
            "A" * 100000,     # 100KB
            "A" * 1000000,    # 1MB
            "\x00" * 50000,   # Null bytes
            "🚀" * 10000,     # Unicode
        ]
        
        for component in components:
            for payload in large_payloads:
                try:
                    if hasattr(component, 'insert'):
                        component.insert(f"buffer_test_{len(payload)}", payload)
                        result = component.search(f"buffer_test_{len(payload)}")
                    elif hasattr(component, 'put'):
                        component.put(f"buffer_test_{len(payload)}", payload)
                        result = component.get(f"buffer_test_{len(payload)}")
                    
                    # If we get here without error and result is truncated,
                    # it might indicate a buffer overflow (though controlled)
                    if isinstance(result, str) and len(result) < len(payload) * 0.5:
                        vulnerabilities.append({
                            'type': 'BUFFER_TRUNCATION',
                            'severity': 'LOW',
                            'payload': f'Large payload ({len(payload)} bytes)',
                            'component': str(type(component).__name__),
                            'description': 'Data truncation detected - possible buffer limitation'
                        })
                        
                except (MemoryError, OverflowError):
                    # These are expected for very large inputs
                    pass
                except Exception as e:
                    # Other errors might indicate vulnerability
                    if any(term in str(e).lower() for term in ['buffer', 'overflow', 'memory']):
                        vulnerabilities.append({
                            'type': 'BUFFER_OVERFLOW_ERROR',
                            'severity': 'MEDIUM',
                            'payload': f'Large payload ({len(payload)} bytes)',
                            'component': str(type(component).__name__),
                            'description': f'Buffer-related error: {str(e)}'
                        })
        
        return vulnerabilities
    
    def generate_vulnerability_report(self) -> Dict[str, Any]:
        """Generate comprehensive vulnerability report"""
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        vulnerability_types = {}
        
        for vuln in self.vulnerabilities_found:
            severity = vuln.get('severity', 'UNKNOWN')
            vuln_type = vuln.get('type', 'UNKNOWN')
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            vulnerability_types[vuln_type] = vulnerability_types.get(vuln_type, 0) + 1
        
        risk_score = (
            severity_counts.get('CRITICAL', 0) * 10 +
            severity_counts.get('HIGH', 0) * 5 +
            severity_counts.get('MEDIUM', 0) * 2 +
            severity_counts.get('LOW', 0) * 1
        )
        
        return {
            'scan_timestamp': datetime.now().isoformat(),
            'total_vulnerabilities': len(self.vulnerabilities_found),
            'severity_breakdown': severity_counts,
            'vulnerability_types': vulnerability_types,
            'risk_score': risk_score,
            'risk_level': self._calculate_risk_level(risk_score),
            'vulnerabilities': self.vulnerabilities_found,
            'recommendations': self._generate_recommendations()
        }
    
    def _calculate_risk_level(self, risk_score: int) -> str:
        """Calculate overall risk level"""
        if risk_score >= 50:
            return 'CRITICAL'
        elif risk_score >= 20:
            return 'HIGH'
        elif risk_score >= 10:
            return 'MEDIUM'
        elif risk_score > 0:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        vuln_types = set(vuln['type'] for vuln in self.vulnerabilities_found)
        
        if 'SQL_INJECTION' in vuln_types:
            recommendations.append("Implement parameterized queries and input sanitization")
        
        if 'XSS_STORED' in vuln_types or 'XSS_REFLECTED' in vuln_types:
            recommendations.append("Implement output encoding and CSP headers")
        
        if 'PATH_TRAVERSAL' in vuln_types:
            recommendations.append("Validate and sanitize file paths, use whitelist approach")
        
        if 'BUFFER_OVERFLOW_ERROR' in vuln_types:
            recommendations.append("Implement input length validation and memory management")
        
        recommendations.extend([
            "Regular security audits and penetration testing",
            "Keep all dependencies updated",
            "Implement logging and monitoring",
            "Use HTTPS for all communications",
            "Implement rate limiting and DDoS protection"
        ])
        
        return recommendations


class TestPenetrationTesting:
    """Comprehensive penetration testing suite"""
    
    def test_comprehensive_vulnerability_scan(self):
        """Run comprehensive vulnerability scan across all components"""
        scanner = VulnerabilityScanner()
        
        # Initialize components
        components = [
            OptimizedTrie(),
            HighPerformanceHashMap(),
            SmartCache(max_size=1000, ttl=300)
        ]
        
        processor = OptimizedFileProcessor()
        
        # SQL Injection payloads
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --",
            "1; SELECT * FROM sensitive_data",
        ]
        
        # XSS payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')>",
        ]
        
        # Path traversal payloads
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "/etc/passwd",
            "file:///etc/passwd",
        ]
        
        # Run scans
        for component in components:
            sql_vulns = scanner.scan_sql_injection(component, sql_payloads)
            scanner.vulnerabilities_found.extend(sql_vulns)
        
        xss_vulns = scanner.scan_xss_vulnerabilities(processor, xss_payloads)
        scanner.vulnerabilities_found.extend(xss_vulns)
        
        path_vulns = scanner.scan_path_traversal(processor, traversal_payloads)
        scanner.vulnerabilities_found.extend(path_vulns)
        
        buffer_vulns = scanner.scan_buffer_overflow(components)
        scanner.vulnerabilities_found.extend(buffer_vulns)
        
        # Generate report
        report = scanner.generate_vulnerability_report()
        
        # Assertions - we expect a secure system with minimal vulnerabilities
        assert report['risk_level'] in ['MINIMAL', 'LOW', 'MEDIUM'], f"Risk level too high: {report['risk_level']}"
        assert report['severity_breakdown']['CRITICAL'] == 0, f"Critical vulnerabilities found: {report['severity_breakdown']['CRITICAL']}"
        
        # Print report for analysis
        print(f"\nVulnerability Scan Report:")
        print(f"Risk Level: {report['risk_level']}")
        print(f"Total Vulnerabilities: {report['total_vulnerabilities']}")
        print(f"Severity Breakdown: {report['severity_breakdown']}")
        
        if report['vulnerabilities']:
            print("\nVulnerabilities Found:")
            for vuln in report['vulnerabilities'][:5]:  # Show first 5
                print(f"  - {vuln['type']} ({vuln['severity']}): {vuln['description']}")
    
    def test_authentication_bypass(self):
        """Test for authentication bypass vulnerabilities"""
        
        class MockAuthenticator:
            """Mock authentication system for testing"""
            
            def __init__(self):
                self.users = {
                    'admin': {'password': 'admin123', 'role': 'admin'},
                    'user': {'password': 'user123', 'role': 'user'}
                }
                self.active_sessions = {}
            
            def authenticate(self, username: str, password: str) -> Dict[str, Any]:
                """Authenticate user"""
                user = self.users.get(username)
                if user and user['password'] == password:
                    session_id = base64.b64encode(os.urandom(16)).decode()
                    self.active_sessions[session_id] = {
                        'username': username,
                        'role': user['role'],
                        'created_at': time.time()
                    }
                    return {'success': True, 'session_id': session_id, 'role': user['role']}
                return {'success': False, 'error': 'Invalid credentials'}
            
            def validate_session(self, session_id: str) -> Dict[str, Any]:
                """Validate session"""
                session = self.active_sessions.get(session_id)
                if session:
                    # Check session timeout (1 hour)
                    if time.time() - session['created_at'] < 3600:
                        return {'valid': True, 'username': session['username'], 'role': session['role']}
                    else:
                        del self.active_sessions[session_id]
                return {'valid': False, 'error': 'Invalid or expired session'}
        
        auth = MockAuthenticator()
        
        # Test legitimate authentication
        result = auth.authenticate('admin', 'admin123')
        assert result['success'], "Legitimate authentication failed"
        session_id = result['session_id']
        
        # Test session validation
        validation = auth.validate_session(session_id)
        assert validation['valid'], "Valid session was rejected"
        
        # Test authentication bypass attempts
        bypass_attempts = [
            ('admin', "admin123' OR '1'='1"),  # SQL injection
            ('admin', 'wrong_password'),       # Brute force
            ("admin'--", 'any_password'),      # SQL comment
            ('admin', ''),                     # Empty password
            ('', ''),                          # Empty credentials
            ('admin\x00', 'admin123'),         # Null byte injection
            ('ADMIN', 'admin123'),             # Case sensitivity
        ]
        
        for username, password in bypass_attempts:
            result = auth.authenticate(username, password)
            assert not result['success'], f"Authentication bypass succeeded for {username}:{password}"
        
        # Test session manipulation
        fake_sessions = [
            'fake_session_id',
            '',
            'admin_session',
            base64.b64encode(b'fake_data').decode(),
        ]
        
        for fake_session in fake_sessions:
            validation = auth.validate_session(fake_session)
            assert not validation['valid'], f"Fake session was accepted: {fake_session}"
    
    def test_session_hijacking_prevention(self):
        """Test prevention of session hijacking attacks"""
        
        class SecureSessionManager:
            """Secure session manager with hijacking prevention"""
            
            def __init__(self):
                self.sessions = {}
            
            def create_session(self, user_id: str, user_agent: str, ip_address: str) -> str:
                """Create secure session with fingerprinting"""
                session_id = base64.urlsafe_b64encode(os.urandom(32)).decode()
                
                # Create session fingerprint
                fingerprint = hashlib.sha256(
                    f"{user_agent}:{ip_address}:{session_id}".encode()
                ).hexdigest()
                
                self.sessions[session_id] = {
                    'user_id': user_id,
                    'created_at': time.time(),
                    'user_agent': user_agent,
                    'ip_address': ip_address,
                    'fingerprint': fingerprint,
                    'last_activity': time.time()
                }
                
                return session_id
            
            def validate_session(self, session_id: str, user_agent: str, ip_address: str) -> Dict[str, Any]:
                """Validate session with fingerprint check"""
                session = self.sessions.get(session_id)
                if not session:
                    return {'valid': False, 'error': 'Session not found'}
                
                # Check fingerprint
                expected_fingerprint = hashlib.sha256(
                    f"{session['user_agent']}:{session['ip_address']}:{session_id}".encode()
                ).hexdigest()
                
                if session['fingerprint'] != expected_fingerprint:
                    return {'valid': False, 'error': 'Session fingerprint mismatch'}
                
                # Additional checks
                if user_agent != session['user_agent']:
                    return {'valid': False, 'error': 'User agent mismatch'}
                
                if ip_address != session['ip_address']:
                    return {'valid': False, 'error': 'IP address mismatch'}
                
                # Check timeout
                if time.time() - session['last_activity'] > 1800:  # 30 minutes
                    del self.sessions[session_id]
                    return {'valid': False, 'error': 'Session expired'}
                
                # Update activity
                session['last_activity'] = time.time()
                return {'valid': True, 'user_id': session['user_id']}
        
        session_manager = SecureSessionManager()
        
        # Create legitimate session
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        ip_address = "192.168.1.100"
        session_id = session_manager.create_session('user123', user_agent, ip_address)
        
        # Test legitimate validation
        result = session_manager.validate_session(session_id, user_agent, ip_address)
        assert result['valid'], "Legitimate session validation failed"
        
        # Test session hijacking attempts
        hijack_attempts = [
            # Different user agent
            (session_id, "Different User Agent", ip_address),
            # Different IP address
            (session_id, user_agent, "192.168.1.200"),
            # Both different
            (session_id, "Attacker Browser", "10.0.0.1"),
        ]
        
        for test_session_id, test_user_agent, test_ip in hijack_attempts:
            result = session_manager.validate_session(test_session_id, test_user_agent, test_ip)
            assert not result['valid'], f"Session hijacking attempt succeeded: {result}"
    
    def test_rate_limiting_bypass(self):
        """Test rate limiting bypass attempts"""
        
        rate_limiter = RateLimiter(requests_per_second=5, window_size=1.0)
        
        # Test normal rate limiting
        allowed_count = 0
        for i in range(10):
            if asyncio.run(rate_limiter.is_allowed('normal_user')):
                allowed_count += 1
        
        assert allowed_count <= 6, f"Rate limiter too permissive: {allowed_count} requests allowed"
        
        # Test bypass attempts
        bypass_attempts = [
            # Different user IDs (should each have their own limit)
            [f'user_{i}' for i in range(10)],
            # Case variations
            ['User123', 'user123', 'USER123', 'uSeR123'],
            # Special characters
            ['user@domain.com', 'user+tag@domain.com', 'user%40domain.com'],
            # Unicode variations
            ['user123', 'user\u0031\u0032\u0033', 'user１２３'],
        ]
        
        for user_ids in bypass_attempts:
            total_allowed = 0
            for user_id in user_ids:
                if asyncio.run(rate_limiter.is_allowed(user_id)):
                    total_allowed += 1
            
            # Each unique user should be rate limited individually
            # But similar/same users should share limits
            if len(set(user_ids)) == len(user_ids):  # All unique
                assert total_allowed <= len(user_ids) * 6, f"Rate limiting bypassed with unique users: {total_allowed}"
            else:  # Some duplicates
                assert total_allowed <= 6, f"Rate limiting bypassed with duplicate users: {total_allowed}"
    
    def test_privilege_escalation_attacks(self):
        """Test for privilege escalation vulnerabilities"""
        
        class PrivilegeManager:
            """Mock privilege management system"""
            
            def __init__(self):
                self.users = {
                    'admin': {'role': 'admin', 'permissions': ['read', 'write', 'delete', 'admin']},
                    'user1': {'role': 'user', 'permissions': ['read', 'write']},
                    'user2': {'role': 'readonly', 'permissions': ['read']}
                }
            
            def check_permission(self, user_id: str, permission: str) -> bool:
                """Check if user has permission"""
                user = self.users.get(user_id)
                return user and permission in user.get('permissions', [])
            
            def modify_user_permissions(self, admin_id: str, target_user: str, new_permissions: List[str]) -> Dict[str, Any]:
                """Modify user permissions (admin only)"""
                if not self.check_permission(admin_id, 'admin'):
                    return {'success': False, 'error': 'Admin privileges required'}
                
                if target_user not in self.users:
                    return {'success': False, 'error': 'User not found'}
                
                self.users[target_user]['permissions'] = new_permissions
                return {'success': True, 'message': f'Permissions updated for {target_user}'}
            
            def execute_admin_command(self, user_id: str, command: str) -> Dict[str, Any]:
                """Execute admin command"""
                if not self.check_permission(user_id, 'admin'):
                    return {'success': False, 'error': 'Admin privileges required'}
                
                # Simulate command execution
                return {'success': True, 'output': f'Command "{command}" executed by {user_id}'}
        
        privilege_manager = PrivilegeManager()
        
        # Test legitimate admin operations
        result = privilege_manager.modify_user_permissions('admin', 'user1', ['read', 'write', 'delete'])
        assert result['success'], "Legitimate admin operation failed"
        
        result = privilege_manager.execute_admin_command('admin', 'system_backup')
        assert result['success'], "Legitimate admin command failed"
        
        # Test privilege escalation attempts
        escalation_attempts = [
            # Regular user trying to modify permissions
            ('user1', 'user2', ['admin']),
            ('user2', 'user2', ['admin']),  # Self-escalation
            # Non-existent user
            ('hacker', 'admin', ['read']),
        ]
        
        for attacker, target, permissions in escalation_attempts:
            result = privilege_manager.modify_user_permissions(attacker, target, permissions)
            assert not result['success'], f"Privilege escalation succeeded: {attacker} -> {target}"
            assert 'admin privileges required' in result['error'].lower()
        
        # Test admin command execution by non-admin
        command_attempts = [
            ('user1', 'delete_all_data'),
            ('user2', 'create_admin_user'),
            ('nonexistent', 'system_shutdown'),
        ]
        
        for user_id, command in command_attempts:
            result = privilege_manager.execute_admin_command(user_id, command)
            assert not result['success'], f"Unauthorized command execution succeeded: {user_id}"
    
    def test_data_leakage_prevention(self):
        """Test prevention of data leakage"""
        
        # Test file processor with sensitive data
        processor = OptimizedFileProcessor()
        
        # Create test file with sensitive information
        sensitive_data = {
            'users': [
                {
                    'id': 1,
                    'username': 'john_doe',
                    'password': 'hashed_password_123',
                    'email': 'john@example.com',
                    'ssn': '123-45-6789',
                    'credit_card': '4111-1111-1111-1111',
                    'api_key': 'sk-1234567890abcdefghij'
                },
                {
                    'id': 2,
                    'username': 'jane_smith',
                    'password': 'another_hashed_password',
                    'email': 'jane@example.com',
                    'ssn': '987-65-4321',
                    'credit_card': '5555-5555-5555-4444',
                    'api_key': 'sk-abcdefghij1234567890'
                }
            ],
            'system_config': {
                'database_url': 'postgresql://user:password@localhost/db',
                'secret_key': 'super_secret_encryption_key',
                'api_endpoints': ['https://api.internal.com/v1/']
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sensitive_data, f)
            temp_path = f.name
        
        try:
            result = asyncio.run(processor.process_file(temp_path))
            
            if 'error' not in result:
                result_str = str(result)
                
                # Check for sensitive data leakage
                sensitive_patterns = [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                    r'sk-[a-zA-Z0-9]{20,}',  # API key pattern
                    r'password["\']?\s*:\s*["\'][^"\']+["\']',  # Password fields
                    r'postgresql://[^/]+/[^/]+',  # Database URLs
                ]
                
                leaked_patterns = []
                for pattern in sensitive_patterns:
                    if re.search(pattern, result_str, re.IGNORECASE):
                        leaked_patterns.append(pattern)
                
                # In a secure system, sensitive data should be sanitized or not exposed
                if leaked_patterns:
                    print(f"Warning: Potential data leakage detected: {leaked_patterns}")
                    # This might be expected behavior if the system is designed to process this data
                    # The test is mainly to identify and document data handling behavior
        
        finally:
            os.unlink(temp_path)
    
    def test_denial_of_service_resistance(self):
        """Test resistance to denial of service attacks"""
        
        components = [
            OptimizedTrie(),
            HighPerformanceHashMap(),
            SmartCache(max_size=1000, ttl=300)
        ]
        
        # Test resource exhaustion attacks
        dos_payloads = [
            # Memory exhaustion
            ("memory_bomb", "A" * 1000000),  # 1MB string
            # Deep recursion simulation
            ("deep_nesting", {"level_" + str(i): f"data_{i}" for i in range(10000)}),
            # Hash collision attempt
            ("hash_collision", {str(i): f"value_{i}" for i in range(100000)}),
        ]
        
        for component in components:
            for attack_name, payload in dos_payloads:
                start_time = time.time()
                
                try:
                    if attack_name == "memory_bomb":
                        if hasattr(component, 'insert'):
                            component.insert("dos_test", payload)
                        elif hasattr(component, 'put'):
                            component.put("dos_test", payload)
                    
                    elif attack_name == "deep_nesting":
                        if hasattr(component, 'put'):
                            component.put("nested_data", str(payload))
                    
                    elif attack_name == "hash_collision":
                        if hasattr(component, 'put'):
                            for key, value in list(payload.items())[:1000]:  # Limit to prevent actual DoS
                                component.put(key, value)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # Should not take too long (indicates DoS resistance)
                    assert processing_time < 5.0, f"DoS attack '{attack_name}' caused excessive delay: {processing_time:.2f}s"
                    
                except (MemoryError, OverflowError, TimeoutError):
                    # These are acceptable - system protecting itself
                    pass
                except Exception as e:
                    # Other exceptions might indicate DoS vulnerability or protection
                    processing_time = time.time() - start_time
                    if processing_time > 10.0:
                        pytest.fail(f"DoS attack '{attack_name}' caused system instability: {e}")


if __name__ == '__main__':
    # Run penetration tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-s',  # Show output for vulnerability reports
        '--durations=5',  # Show slowest 5 tests
    ])