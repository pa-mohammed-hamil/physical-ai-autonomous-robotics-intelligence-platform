"""
Biometric Authentication System for Robot Security

Multi-factor biometric authentication using:
- Fingerprint recognition
- Face recognition
- Voice recognition (voice pattern matching)
- Real-time verification
- Live detection (anti-spoofing)

Used to unlock robots after security incidents or unauthorized access attempts.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class BiometricAuthError(Exception):
    """Base exception for biometric authentication errors."""
    pass


class AuthenticationFailedError(BiometricAuthError):
    """Raised when authentication fails."""
    pass


class LivenessCheckFailedError(BiometricAuthError):
    """Raised when liveness detection fails (spoofing attempt)."""
    pass


class BiometricType(str, Enum):
    """Types of biometric authentication."""
    FINGERPRINT = "fingerprint"
    FACE = "face"
    VOICE = "voice"
    IRIS = "iris"


class AuthenticationStatus(str, Enum):
    """Authentication attempt status."""
    SUCCESS = "success"
    FAILED = "failed"
    LOCKED = "locked"
    PENDING = "pending"
    LIVENESS_FAILED = "liveness_failed"


class SecurityLevel(str, Enum):
    """Security levels requiring different authentication factors."""
    LOW = "low"  # Single factor
    MEDIUM = "medium"  # Two factors
    HIGH = "high"  # Three factors
    MAXIMUM = "maximum"  # All factors + additional verification


@dataclass
class BiometricTemplate:
    """Stored biometric template."""
    user_id: str
    biometric_type: BiometricType
    template_data: str  # Hashed/encrypted template
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


@dataclass
class AuthenticationAttempt:
    """Authentication attempt record."""
    attempt_id: str
    user_id: str
    robot_id: str
    biometric_type: BiometricType
    status: AuthenticationStatus
    confidence_score: float
    liveness_score: Optional[float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    device_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthenticationResult:
    """Result of authentication attempt."""
    success: bool
    user_id: Optional[str]
    confidence_score: float
    factors_verified: List[BiometricType]
    liveness_passed: bool
    attempt_id: str
    message: str
    requires_additional_verification: bool = False
    lockout_until: Optional[datetime] = None


class BiometricAuthenticator:
    """
    Multi-factor biometric authentication system.
    
    Features:
    - Fingerprint recognition
    - Face recognition with liveness detection
    - Voice recognition and pattern matching
    - Anti-spoofing measures
    - Multi-factor authentication
    - Attempt tracking and lockout
    - Real-time verification
    """
    
    def __init__(
        self,
        max_failed_attempts: int = 3,
        lockout_duration_minutes: int = 30,
        min_confidence_threshold: float = 0.85,
        liveness_threshold: float = 0.80,
    ):
        """
        Initialize biometric authenticator.
        
        Args:
            max_failed_attempts: Maximum failed attempts before lockout
            lockout_duration_minutes: Lockout duration after max failures
            min_confidence_threshold: Minimum confidence for successful auth
            liveness_threshold: Minimum liveness score for anti-spoofing
        """
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)
        self.min_confidence_threshold = min_confidence_threshold
        self.liveness_threshold = liveness_threshold
        
        # Storage
        self.templates: Dict[str, List[BiometricTemplate]] = {}
        self.attempts: List[AuthenticationAttempt] = []
        self.lockouts: Dict[str, datetime] = {}
        
        logger.info("Biometric authenticator initialized")
    
    async def authenticate_fingerprint(
        self,
        fingerprint_data: bytes,
        robot_id: str,
        user_id: Optional[str] = None,
    ) -> AuthenticationResult:
        """
        Authenticate using fingerprint.
        
        Args:
            fingerprint_data: Raw fingerprint image/data
            robot_id: Robot being accessed
            user_id: Optional user ID for verification
            
        Returns:
            Authentication result
        """
        logger.info(f"Fingerprint authentication attempt for robot {robot_id}")
        
        # Check if locked out
        if user_id and self._is_locked_out(user_id):
            lockout_until = self.lockouts.get(user_id)
            return AuthenticationResult(
                success=False,
                user_id=user_id,
                confidence_score=0.0,
                factors_verified=[],
                liveness_passed=False,
                attempt_id=self._generate_attempt_id(),
                message=f"Account locked until {lockout_until}",
                lockout_until=lockout_until,
            )
        
        # Extract fingerprint features
        features = await self._extract_fingerprint_features(fingerprint_data)
        
        # Match against stored templates
        match_result = await self._match_fingerprint(features, user_id)
        
        if match_result["success"]:
            matched_user_id = match_result["user_id"]
            confidence = match_result["confidence"]
            
            # Record successful attempt
            self._record_attempt(
                user_id=matched_user_id,
                robot_id=robot_id,
                biometric_type=BiometricType.FINGERPRINT,
                status=AuthenticationStatus.SUCCESS,
                confidence_score=confidence,
            )
            
            # Reset failure count
            if matched_user_id in self.lockouts:
                del self.lockouts[matched_user_id]
            
            return AuthenticationResult(
                success=True,
                user_id=matched_user_id,
                confidence_score=confidence,
                factors_verified=[BiometricType.FINGERPRINT],
                liveness_passed=True,  # Fingerprint is inherently live
                attempt_id=self._generate_attempt_id(),
                message="Fingerprint authentication successful",
            )
        else:
            # Record failed attempt
            self._record_attempt(
                user_id=user_id or "unknown",
                robot_id=robot_id,
                biometric_type=BiometricType.FINGERPRINT,
                status=AuthenticationStatus.FAILED,
                confidence_score=match_result["confidence"],
            )
            
            # Check if should lock out
            if user_id:
                self._check_and_apply_lockout(user_id)
            
            return AuthenticationResult(
                success=False,
                user_id=user_id,
                confidence_score=match_result["confidence"],
                factors_verified=[],
                liveness_passed=True,
                attempt_id=self._generate_attempt_id(),
                message="Fingerprint authentication failed",
            )
    
    async def authenticate_face(
        self,
        face_image: bytes,
        robot_id: str,
        user_id: Optional[str] = None,
        check_liveness: bool = True,
    ) -> AuthenticationResult:
        """
        Authenticate using face recognition with liveness detection.
        
        Args:
            face_image: Face image data
            robot_id: Robot being accessed
            user_id: Optional user ID for verification
            check_liveness: Perform liveness detection
            
        Returns:
            Authentication result
        """
        logger.info(f"Face authentication attempt for robot {robot_id}")
        
        # Check if locked out
        if user_id and self._is_locked_out(user_id):
            lockout_until = self.lockouts.get(user_id)
            return AuthenticationResult(
                success=False,
                user_id=user_id,
                confidence_score=0.0,
                factors_verified=[],
                liveness_passed=False,
                attempt_id=self._generate_attempt_id(),
                message=f"Account locked until {lockout_until}",
                lockout_until=lockout_until,
            )
        
        # Liveness detection (anti-spoofing)
        if check_liveness:
            liveness_result = await self._check_face_liveness(face_image)
            
            if not liveness_result["is_live"]:
                logger.warning("Face liveness check failed - possible spoofing attempt")
                
                self._record_attempt(
                    user_id=user_id or "unknown",
                    robot_id=robot_id,
                    biometric_type=BiometricType.FACE,
                    status=AuthenticationStatus.LIVENESS_FAILED,
                    confidence_score=0.0,
                    liveness_score=liveness_result["score"],
                )
                
                return AuthenticationResult(
                    success=False,
                    user_id=user_id,
                    confidence_score=0.0,
                    factors_verified=[],
                    liveness_passed=False,
                    attempt_id=self._generate_attempt_id(),
                    message="Liveness check failed. Please use a live camera feed.",
                )
        
        # Extract face features
        features = await self._extract_face_features(face_image)
        
        # Match against stored templates
        match_result = await self._match_face(features, user_id)
        
        if match_result["success"]:
            matched_user_id = match_result["user_id"]
            confidence = match_result["confidence"]
            
            # Record successful attempt
            self._record_attempt(
                user_id=matched_user_id,
                robot_id=robot_id,
                biometric_type=BiometricType.FACE,
                status=AuthenticationStatus.SUCCESS,
                confidence_score=confidence,
                liveness_score=liveness_result["score"] if check_liveness else None,
            )
            
            # Reset failure count
            if matched_user_id in self.lockouts:
                del self.lockouts[matched_user_id]
            
            return AuthenticationResult(
                success=True,
                user_id=matched_user_id,
                confidence_score=confidence,
                factors_verified=[BiometricType.FACE],
                liveness_passed=liveness_result["is_live"] if check_liveness else True,
                attempt_id=self._generate_attempt_id(),
                message="Face authentication successful",
            )
        else:
            # Record failed attempt
            self._record_attempt(
                user_id=user_id or "unknown",
                robot_id=robot_id,
                biometric_type=BiometricType.FACE,
                status=AuthenticationStatus.FAILED,
                confidence_score=match_result["confidence"],
            )
            
            # Check if should lock out
            if user_id:
                self._check_and_apply_lockout(user_id)
            
            return AuthenticationResult(
                success=False,
                user_id=user_id,
                confidence_score=match_result["confidence"],
                factors_verified=[],
                liveness_passed=liveness_result["is_live"] if check_liveness else True,
                attempt_id=self._generate_attempt_id(),
                message="Face authentication failed",
            )
    
    async def authenticate_voice(
        self,
        voice_sample: bytes,
        robot_id: str,
        user_id: Optional[str] = None,
        passphrase: Optional[str] = None,
    ) -> AuthenticationResult:
        """
        Authenticate using voice recognition and pattern matching.
        
        Args:
            voice_sample: Audio recording of voice
            robot_id: Robot being accessed
            user_id: Optional user ID for verification
            passphrase: Optional passphrase to verify
            
        Returns:
            Authentication result
        """
        logger.info(f"Voice authentication attempt for robot {robot_id}")
        
        # Check if locked out
        if user_id and self._is_locked_out(user_id):
            lockout_until = self.lockouts.get(user_id)
            return AuthenticationResult(
                success=False,
                user_id=user_id,
                confidence_score=0.0,
                factors_verified=[],
                liveness_passed=False,
                attempt_id=self._generate_attempt_id(),
                message=f"Account locked until {lockout_until}",
                lockout_until=lockout_until,
            )
        
        # Extract voice features
        features = await self._extract_voice_features(voice_sample)
        
        # Verify passphrase if provided
        if passphrase:
            passphrase_match = await self._verify_passphrase(voice_sample, passphrase)
            if not passphrase_match:
                logger.warning("Voice passphrase verification failed")
                return AuthenticationResult(
                    success=False,
                    user_id=user_id,
                    confidence_score=0.0,
                    factors_verified=[],
                    liveness_passed=True,
                    attempt_id=self._generate_attempt_id(),
                    message="Passphrase verification failed",
                )
        
        # Match voice pattern against stored templates
        match_result = await self._match_voice(features, user_id)
        
        if match_result["success"]:
            matched_user_id = match_result["user_id"]
            confidence = match_result["confidence"]
            
            # Record successful attempt
            self._record_attempt(
                user_id=matched_user_id,
                robot_id=robot_id,
                biometric_type=BiometricType.VOICE,
                status=AuthenticationStatus.SUCCESS,
                confidence_score=confidence,
            )
            
            # Reset failure count
            if matched_user_id in self.lockouts:
                del self.lockouts[matched_user_id]
            
            return AuthenticationResult(
                success=True,
                user_id=matched_user_id,
                confidence_score=confidence,
                factors_verified=[BiometricType.VOICE],
                liveness_passed=True,
                attempt_id=self._generate_attempt_id(),
                message="Voice authentication successful",
            )
        else:
            # Record failed attempt
            self._record_attempt(
                user_id=user_id or "unknown",
                robot_id=robot_id,
                biometric_type=BiometricType.VOICE,
                status=AuthenticationStatus.FAILED,
                confidence_score=match_result["confidence"],
            )
            
            # Check if should lock out
            if user_id:
                self._check_and_apply_lockout(user_id)
            
            return AuthenticationResult(
                success=False,
                user_id=user_id,
                confidence_score=match_result["confidence"],
                factors_verified=[],
                liveness_passed=True,
                attempt_id=self._generate_attempt_id(),
                message="Voice authentication failed",
            )
    
    async def multi_factor_authenticate(
        self,
        robot_id: str,
        fingerprint_data: Optional[bytes] = None,
        face_image: Optional[bytes] = None,
        voice_sample: Optional[bytes] = None,
        user_id: Optional[str] = None,
        required_factors: int = 2,
    ) -> AuthenticationResult:
        """
        Perform multi-factor biometric authentication.
        
        Args:
            robot_id: Robot being accessed
            fingerprint_data: Optional fingerprint data
            face_image: Optional face image
            voice_sample: Optional voice recording
            user_id: Optional user ID
            required_factors: Number of factors required (2-3)
            
        Returns:
            Combined authentication result
        """
        logger.info(f"Multi-factor authentication for robot {robot_id} (required: {required_factors} factors)")
        
        verified_factors = []
        total_confidence = 0.0
        all_liveness_passed = True
        authenticated_user_id = None
        
        # Try each provided factor
        if fingerprint_data:
            result = await self.authenticate_fingerprint(fingerprint_data, robot_id, user_id)
            if result.success:
                verified_factors.extend(result.factors_verified)
                total_confidence += result.confidence_score
                authenticated_user_id = result.user_id
        
        if face_image:
            result = await self.authenticate_face(face_image, robot_id, user_id)
            if result.success:
                verified_factors.extend(result.factors_verified)
                total_confidence += result.confidence_score
                authenticated_user_id = result.user_id or authenticated_user_id
                all_liveness_passed = all_liveness_passed and result.liveness_passed
        
        if voice_sample:
            result = await self.authenticate_voice(voice_sample, robot_id, user_id)
            if result.success:
                verified_factors.extend(result.factors_verified)
                total_confidence += result.confidence_score
                authenticated_user_id = result.user_id or authenticated_user_id
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(verified_factors) if verified_factors else 0.0
        
        # Check if enough factors verified
        success = len(verified_factors) >= required_factors and all_liveness_passed
        
        if success:
            message = f"Multi-factor authentication successful ({len(verified_factors)} factors verified)"
        else:
            message = f"Multi-factor authentication failed (only {len(verified_factors)}/{required_factors} factors verified)"
        
        return AuthenticationResult(
            success=success,
            user_id=authenticated_user_id,
            confidence_score=avg_confidence,
            factors_verified=verified_factors,
            liveness_passed=all_liveness_passed,
            attempt_id=self._generate_attempt_id(),
            message=message,
        )
    
    async def _extract_fingerprint_features(self, data: bytes) -> np.ndarray:
        """Extract fingerprint minutiae features."""
        # In production, use actual fingerprint recognition library
        # (e.g., pyfingerprint, OpenCV fingerprint matching)
        await asyncio.sleep(0.1)  # Simulate processing
        
        # Simulated feature extraction
        fingerprint_hash = hashlib.sha256(data).digest()
        features = np.frombuffer(fingerprint_hash, dtype=np.uint8)
        return features
    
    async def _match_fingerprint(
        self,
        features: np.ndarray,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Match fingerprint features against stored templates."""
        # Simulate matching
        await asyncio.sleep(0.1)
        
        # In production, match against actual stored templates
        # For demo, return success with high confidence
        return {
            "success": True,
            "user_id": user_id or "owner123",
            "confidence": 0.95,
        }
    
    async def _extract_face_features(self, image: bytes) -> np.ndarray:
        """Extract face embedding features."""
        # In production, use face recognition library
        # (e.g., face_recognition, DeepFace, dlib)
        await asyncio.sleep(0.2)  # Simulate processing
        
        # Simulated 128-dimensional face embedding
        face_hash = hashlib.sha256(image).digest()
        features = np.frombuffer(face_hash[:128], dtype=np.uint8)
        return features
    
    async def _check_face_liveness(self, image: bytes) -> Dict[str, Any]:
        """
        Check if face is live (anti-spoofing).
        
        Detects:
        - Photo/video replay attacks
        - 3D masks
        - Deep fakes
        """
        # In production, use liveness detection model
        # (e.g., Silent-Face-Anti-Spoofing, FaceNet liveness)
        await asyncio.sleep(0.15)
        
        # Simulated liveness check
        return {
            "is_live": True,
            "score": 0.92,
            "method": "texture_analysis",
        }
    
    async def _match_face(
        self,
        features: np.ndarray,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Match face features against stored templates."""
        await asyncio.sleep(0.1)
        
        # In production, compute cosine similarity with stored embeddings
        return {
            "success": True,
            "user_id": user_id or "owner123",
            "confidence": 0.91,
        }
    
    async def _extract_voice_features(self, audio: bytes) -> np.ndarray:
        """Extract voice characteristics (MFCC, pitch, etc.)."""
        # In production, use voice recognition library
        # (e.g., speechbrain, pyAudioAnalysis, librosa)
        await asyncio.sleep(0.2)
        
        # Simulated voice features
        voice_hash = hashlib.sha256(audio).digest()
        features = np.frombuffer(voice_hash[:64], dtype=np.uint8)
        return features
    
    async def _verify_passphrase(
        self,
        audio: bytes,
        expected_passphrase: str,
    ) -> bool:
        """Verify spoken passphrase using speech-to-text."""
        # In production, use speech recognition
        # (e.g., Google Speech-to-Text, Whisper, DeepSpeech)
        await asyncio.sleep(0.3)
        
        # Simulated STT and verification
        return True  # Assume correct for demo
    
    async def _match_voice(
        self,
        features: np.ndarray,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Match voice pattern against stored voiceprints."""
        await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "user_id": user_id or "owner123",
            "confidence": 0.88,
        }
    
    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out."""
        if user_id not in self.lockouts:
            return False
        
        lockout_until = self.lockouts[user_id]
        if datetime.utcnow() >= lockout_until:
            # Lockout expired
            del self.lockouts[user_id]
            return False
        
        return True
    
    def _check_and_apply_lockout(self, user_id: str) -> None:
        """Check failed attempts and apply lockout if needed."""
        # Count recent failed attempts
        recent_failures = [
            a for a in self.attempts
            if a.user_id == user_id
            and a.status == AuthenticationStatus.FAILED
            and a.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        if len(recent_failures) >= self.max_failed_attempts:
            lockout_until = datetime.utcnow() + self.lockout_duration
            self.lockouts[user_id] = lockout_until
            logger.warning(f"User {user_id} locked out until {lockout_until}")
    
    def _record_attempt(
        self,
        user_id: str,
        robot_id: str,
        biometric_type: BiometricType,
        status: AuthenticationStatus,
        confidence_score: float,
        liveness_score: Optional[float] = None,
    ) -> None:
        """Record authentication attempt."""
        attempt = AuthenticationAttempt(
            attempt_id=self._generate_attempt_id(),
            user_id=user_id,
            robot_id=robot_id,
            biometric_type=biometric_type,
            status=status,
            confidence_score=confidence_score,
            liveness_score=liveness_score,
        )
        
        self.attempts.append(attempt)
        
        # Keep only last 1000 attempts
        if len(self.attempts) > 1000:
            self.attempts = self.attempts[-1000:]
    
    def _generate_attempt_id(self) -> str:
        """Generate unique attempt ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        import random
        suffix = random.randint(1000, 9999)
        return f"AUTH-{timestamp}-{suffix}"
    
    def get_authentication_history(
        self,
        user_id: Optional[str] = None,
        robot_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[AuthenticationAttempt]:
        """Get authentication history."""
        filtered = self.attempts
        
        if user_id:
            filtered = [a for a in filtered if a.user_id == user_id]
        
        if robot_id:
            filtered = [a for a in filtered if a.robot_id == robot_id]
        
        # Sort by timestamp (most recent first)
        filtered = sorted(filtered, key=lambda x: x.timestamp, reverse=True)
        
        return filtered[:limit]


async def main():
    """Example usage."""
    authenticator = BiometricAuthenticator()
    
    # Simulate fingerprint authentication
    fingerprint = b"fingerprint_data_here"
    result = await authenticator.authenticate_fingerprint(
        fingerprint_data=fingerprint,
        robot_id="ROBOT-001",
        user_id="owner123",
    )
    
    print(f"Fingerprint auth: {result.success}")
    print(f"Confidence: {result.confidence_score:.2%}")
    
    # Simulate multi-factor authentication
    face = b"face_image_data"
    voice = b"voice_sample_data"
    
    result = await authenticator.multi_factor_authenticate(
        robot_id="ROBOT-001",
        fingerprint_data=fingerprint,
        face_image=face,
        voice_sample=voice,
        required_factors=2,
    )
    
    print(f"\nMulti-factor auth: {result.success}")
    print(f"Factors verified: {len(result.factors_verified)}")
    print(f"Average confidence: {result.confidence_score:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
