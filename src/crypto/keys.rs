use std::sync::Arc;

use sha2::Digest;

use crate::TopicId;

/// Trait for deriving time-rotated encryption keys.
///
/// Implementations control how encryption keys rotate based on time,
/// providing key isolation across time slots.
pub trait SecretRotation: Send + Sync {
    /// Derive an encryption key for a specific time slot.
    ///
    /// # Arguments
    ///
    /// * `topic_hash` - 32-byte topic identifier
    /// * `unix_minute` - Time slot (minute precision)
    /// * `initial_secret_hash` - 32-byte hashed initial secret
    ///
    /// # Returns
    ///
    /// A 32-byte derived key unique to this topic/time combination.
    fn derive(
        &self,
        topic_hash: [u8; 32],
        unix_minute: u64,
        initial_secret_hash: [u8; 32],
    ) -> [u8; 32];
}

/// Default implementation: SHA512-based KDF.
///
/// Combines topic hash, time slot, and initial secret into a unique key.
#[derive(Debug, Clone)]
pub struct DefaultSecretRotation;

impl SecretRotation for DefaultSecretRotation {
    fn derive(
        &self,
        topic_hash: [u8; 32],
        unix_minute: u64,
        initial_secret_hash: [u8; 32],
    ) -> [u8; 32] {
        use sha2::Digest;
        let mut h = sha2::Sha512::new();
        h.update(topic_hash);
        h.update(unix_minute.to_be_bytes());
        h.update(initial_secret_hash);
        h.finalize()[..32]
            .try_into()
            .expect("keys -> SecretRotation.derive() hash try into [..32] failed")
    }
}

/// Wrapper for custom or default secret rotation implementations.
///
/// Allows pluggable key derivation strategies while maintaining a consistent API.
#[derive(Clone)]
pub struct RotationHandle(Arc<dyn SecretRotation>);

impl core::fmt::Debug for RotationHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotationHandle").finish()
    }
}

impl Default for RotationHandle {
    fn default() -> Self {
        Self(Arc::new(DefaultSecretRotation))
    }
}

impl RotationHandle {
    /// Create a new rotation handle with a custom implementation.
    pub fn new(rotation: impl SecretRotation + 'static) -> Self {
        Self(Arc::new(rotation))
    }

    /// Derive a key using the underlying strategy.
    pub fn derive(
        &self,
        topic_hash: [u8; 32],
        unix_minute: u64,
        initial_secret_hash: [u8; 32],
    ) -> [u8; 32] {
        self.0.derive(topic_hash, unix_minute, initial_secret_hash)
    }
}

/// Derive Ed25519 signing key for DHT record authentication.
///
/// Keys are deterministic per topic, time slot, and shared secret. All nodes
/// holding the same secret derive the same keypair for a given topic+time
/// combination, and its verifying key serves as the DHT routing key for
/// storing/retrieving bootstrap records. The actual record content is signed
/// separately by each node's individual keypair (not this one).
///
/// # Why the secret is mixed in
///
/// This key is the **write capability** for the topic's DHT location. BEP44
/// mutable items are addressed by `(public key, salt)` and a storing node
/// accepts any `put` carrying a higher sequence number than the one it holds.
/// If this keypair were derived from the topic hash alone, anyone who learned
/// the topic *name* could derive it, publish at a high sequence number, and
/// overwrite every legitimate bootstrap record — without holding the secret
/// that protects record contents. Binding the secret here means the DHT
/// location is unguessable from the name, and only secret-holders can write.
///
/// A topic created with an empty secret keeps the old behaviour: derivable by
/// anyone who knows the name, which is the correct semantics for a public
/// topic.
///
/// # Example
///
/// ```ignore
/// let topic = TopicId::from_str("my-topic")?;
/// let unix_minute = crate::unix_minute(0);
/// let signing_key = signing_keypair(&topic, unix_minute, initial_secret_hash);
/// ```
pub fn signing_keypair(
    topic_id: &TopicId,
    unix_minute: u64,
    initial_secret_hash: [u8; 32],
) -> ed25519_dalek::SigningKey {
    let mut sign_keypair_hash = sha2::Sha512::new();
    sign_keypair_hash.update(topic_id.hash());
    sign_keypair_hash.update(unix_minute.to_le_bytes());
    sign_keypair_hash.update(initial_secret_hash);
    let sign_keypair_seed: [u8; 32] = sign_keypair_hash.finalize()[..32]
        .try_into()
        .expect("hashing failed");
    ed25519_dalek::SigningKey::from_bytes(&sign_keypair_seed)
}

/// Derive Ed25519 key for HPKE encryption/decryption.
///
/// Incorporates the secret rotation strategy for time-slot isolation.
///
/// # Example
///
/// ```ignore
/// let topic = TopicId::from_str("my-topic")?;
/// let rotation = RotationHandle::default();
/// let enc_key = encryption_keypair(&topic, &rotation, initial_hash, 0);
/// ```
pub fn encryption_keypair(
    topic_id: &TopicId,
    secret_rotation_function: &RotationHandle,
    initial_secret_hash: [u8; 32],
    unix_minute: u64,
) -> ed25519_dalek::SigningKey {
    let enc_keypair_seed =
        secret_rotation_function
            .0
            .derive(topic_id.hash(), unix_minute, initial_secret_hash);
    ed25519_dalek::SigningKey::from_bytes(&enc_keypair_seed)
}

/// Derive DHT salt for mutable record lookups.
///
/// Salt = SHA512("salt" || topic_hash || unix_minute.to_le_bytes() ||
/// initial_secret_hash)[..32]
///
/// Ensures records are stored in different DHT slots per minute, and — because
/// the secret is mixed in — that the slot for a private topic cannot be located
/// by someone who only knows the topic name. See [`signing_keypair`] for why
/// that matters.
pub fn salt(topic_id: &TopicId, unix_minute: u64, initial_secret_hash: [u8; 32]) -> [u8; 32] {
    let mut slot_hash = sha2::Sha512::new();
    slot_hash.update(b"salt");
    slot_hash.update(topic_id.hash());
    slot_hash.update(unix_minute.to_le_bytes());
    slot_hash.update(initial_secret_hash);
    slot_hash.finalize()[..32]
        .try_into()
        .expect("hashing failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    fn secret_hash(secret: &[u8]) -> [u8; 32] {
        let mut h = sha2::Sha512::new();
        h.update(secret);
        h.finalize()[..32].try_into().expect("hashing failed")
    }

    /// The DHT location must depend on the SECRET, not just the topic name.
    ///
    /// This is the property the fix exists for. The signing keypair is the
    /// write capability for a topic's DHT slot: BEP44 mutable items are
    /// addressed by `(public key, salt)`, and a storing node accepts any put
    /// carrying a higher sequence number. Derived from the topic hash alone,
    /// anyone who learns the topic *name* can derive that keypair and
    /// overwrite every bootstrap record for the topic — without ever holding
    /// the secret that protects record contents.
    #[test]
    fn location_depends_on_the_secret() {
        let topic = TopicId::new("my-topic".to_string());
        let minute = 29_000_000u64;

        let ours = secret_hash(b"the real secret");
        let theirs = secret_hash(b"guessed from the topic name");

        let our_key = signing_keypair(&topic, minute, ours);
        let their_key = signing_keypair(&topic, minute, theirs);
        assert_ne!(
            our_key.verifying_key().to_bytes(),
            their_key.verifying_key().to_bytes(),
            "someone with only the topic name must not derive our write key"
        );

        assert_ne!(
            salt(&topic, minute, ours),
            salt(&topic, minute, theirs),
            "someone with only the topic name must not locate our DHT slot"
        );
    }

    /// Everyone holding the same secret must land on the same slot, or they
    /// would never find each other.
    #[test]
    fn same_secret_same_location() {
        let topic = TopicId::new("my-topic".to_string());
        let minute = 29_000_000u64;
        let s = secret_hash(b"shared");

        assert_eq!(
            signing_keypair(&topic, minute, s)
                .verifying_key()
                .to_bytes(),
            signing_keypair(&topic, minute, s)
                .verifying_key()
                .to_bytes(),
        );
        assert_eq!(salt(&topic, minute, s), salt(&topic, minute, s));
    }

    /// A public topic is one with an empty secret: still deterministic, and
    /// still derivable by anyone who knows the name. That is the intended
    /// semantics, not an oversight.
    #[test]
    fn empty_secret_is_a_public_topic() {
        let topic = TopicId::new("public-topic".to_string());
        let minute = 29_000_000u64;
        let empty = secret_hash(b"");

        let a = signing_keypair(&topic, minute, empty);
        let b = signing_keypair(&topic, minute, empty);
        assert_eq!(a.verifying_key().to_bytes(), b.verifying_key().to_bytes());
    }

    /// Rotation must still work: a different minute is a different slot.
    #[test]
    fn location_still_rotates_per_minute() {
        let topic = TopicId::new("my-topic".to_string());
        let s = secret_hash(b"shared");

        assert_ne!(
            signing_keypair(&topic, 29_000_000, s)
                .verifying_key()
                .to_bytes(),
            signing_keypair(&topic, 29_000_001, s)
                .verifying_key()
                .to_bytes(),
        );
        assert_ne!(salt(&topic, 29_000_000, s), salt(&topic, 29_000_001, s));
    }

    /// Different topics under the same secret must not collide.
    #[test]
    fn different_topics_do_not_collide() {
        let minute = 29_000_000u64;
        let s = secret_hash(b"shared");
        let a = TopicId::new("topic-a".to_string());
        let b = TopicId::new("topic-b".to_string());

        assert_ne!(
            signing_keypair(&a, minute, s).verifying_key().to_bytes(),
            signing_keypair(&b, minute, s).verifying_key().to_bytes(),
        );
        assert_ne!(salt(&a, minute, s), salt(&b, minute, s));
    }
}
