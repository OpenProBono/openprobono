import os
import uuid

from app import db
from app.models import BotRequest, ChatModelParams, EngineEnum, OpenAIModelEnum, User

TEST_API_KEY = os.environ["OPB_TEST_API_KEY"]

def test_validKey() -> None:
    key = TEST_API_KEY
    assert db.admin_check(key) is False
    assert db.api_key_check(key) is True

def test_invalidKey() -> None:
    key = "abc"
    assert db.admin_check(key) is False
    assert db.api_key_check(key) is False

def test_bot_operations() -> None:
    """Test storing, loading, and browsing bots with different users."""
    # Create test users
    user1 = User(firebase_uid="test_user_1", email="test1@example.com")
    user2 = User(firebase_uid="test_user_2", email="test2@example.com")
    
    # Create test bots for each user
    bot1 = BotRequest(
        name="User1's First Bot",
        system_prompt="Test bot 1 for user 1",
        chat_model=ChatModelParams(
            engine=EngineEnum.openai,
            model=OpenAIModelEnum.gpt_4o_mini.value
        ),
        user=user1
    )
    
    bot2 = BotRequest(
        name="User1's Second Bot",
        system_prompt="Test bot 2 for user 1",
        chat_model=ChatModelParams(
            engine=EngineEnum.openai,
            model=OpenAIModelEnum.gpt_4o_mini.value
        ),
        user=user1
    )
    
    bot3 = BotRequest(
        name="User2's Bot",
        system_prompt="Test bot 3 for user 2",
        chat_model=ChatModelParams(
            engine=EngineEnum.openai,
            model=OpenAIModelEnum.gpt_4o_mini.value
        ),
        user=user2
    )
    
    # Generate unique IDs for the bots
    bot1_id = f"test_bot_{uuid.uuid4()}"
    bot2_id = f"test_bot_{uuid.uuid4()}"
    bot3_id = f"test_bot_{uuid.uuid4()}"
    
    # Create session IDs
    session1_id = f"test_session_{uuid.uuid4()}"  # user1's session with bot1
    session2_id = f"test_session_{uuid.uuid4()}"  # user2's session with bot1 (owned by user1)
    session3_id = f"test_session_{uuid.uuid4()}"  # user1's session with bot3 (owned by user2)
    session4_id = f"test_session_{uuid.uuid4()}"  # user2's session with bot3
    
    try:
        # Test storing bots
        db.store_bot(bot1, bot1_id)
        db.store_bot(bot2, bot2_id)
        db.store_bot(bot3, bot3_id)
        
        # Test loading bots
        loaded_bot1 = db.load_bot(bot1_id)
        loaded_bot2 = db.load_bot(bot2_id)
        loaded_bot3 = db.load_bot(bot3_id)
        
        assert loaded_bot1 is not None
        assert loaded_bot2 is not None
        assert loaded_bot3 is not None
        
        assert loaded_bot1.system_prompt == bot1.system_prompt
        assert loaded_bot2.system_prompt == bot2.system_prompt
        assert loaded_bot3.system_prompt == bot3.system_prompt
        
        assert loaded_bot1.user.firebase_uid == user1.firebase_uid
        assert loaded_bot1.user.email == user1.email
        assert loaded_bot2.user.firebase_uid == user1.firebase_uid
        assert loaded_bot3.user.firebase_uid == user2.firebase_uid
        assert loaded_bot3.user.email == user2.email
        
        # Test browsing bots - user1 should see their own bots
        user1_bots = db.browse_bots(user1)
        assert bot1_id in user1_bots
        assert bot2_id in user1_bots
        assert bot3_id not in user1_bots
        
        # Test browsing bots - user2 should see their own bots
        user2_bots = db.browse_bots(user2)
        assert bot1_id not in user2_bots
        assert bot2_id not in user2_bots
        assert bot3_id in user2_bots
        
        # Set up sessions (normally this would be done through the API)
        db.set_session_to_bot(session1_id, bot1_id)
        db.set_session_to_bot(session2_id, bot1_id)
        db.set_session_to_bot(session3_id, bot3_id)
        db.set_session_to_bot(session4_id, bot3_id)
        
        # Store complete session data with full user objects
        db.db.collection(db.CONVERSATION_COLLECTION + db.DB_VERSION).document(session1_id).set({
            "bot_id": bot1_id,
            "firebase_uid": user1.firebase_uid,
            "user": user1.model_dump(),
            "history": [{"role": "user", "content": "Hello from user1 to bot1"}]
        })
        
        db.db.collection(db.CONVERSATION_COLLECTION + db.DB_VERSION).document(session2_id).set({
            "bot_id": bot1_id,
            "firebase_uid": user2.firebase_uid,
            "user": user2.model_dump(),
            "history": [{"role": "user", "content": "Hello from user2 to bot1"}]
        })
        
        db.db.collection(db.CONVERSATION_COLLECTION + db.DB_VERSION).document(session3_id).set({
            "bot_id": bot3_id,
            "firebase_uid": user1.firebase_uid,
            "user": user1.model_dump(),
            "history": [{"role": "user", "content": "Hello from user1 to bot3"}]
        })
        
        db.db.collection(db.CONVERSATION_COLLECTION + db.DB_VERSION).document(session4_id).set({
            "bot_id": bot3_id,
            "firebase_uid": user2.firebase_uid,
            "user": user2.model_dump(),
            "history": [{"role": "user", "content": "Hello from user2 to bot3"}]
        })
        
        # Test 1: User1 should see their own session for bot1
        user1_sessions = db.fetch_sessions_by(bot_id=bot1_id, firebase_uid=None, user=user1)
        assert len(user1_sessions) >= 1
        
        # Find user1's session with bot1
        user1_bot1_session = None
        for session in user1_sessions:
            if session["firebase_uid"] == user1.firebase_uid:
                user1_bot1_session = session
                break
        
        assert user1_bot1_session is not None
        assert user1_bot1_session["bot_id"] == bot1_id
        assert user1_bot1_session["firebase_uid"] == user1.firebase_uid
        assert "user" in user1_bot1_session
        assert user1_bot1_session["user"]["firebase_uid"] == user1.firebase_uid
        assert user1_bot1_session["user"]["email"] == user1.email
        
        # Test 2: User1 (bot1 owner) should also see user2's session with bot1
        user2_bot1_session = None
        for session in user1_sessions:
            if session["firebase_uid"] == user2.firebase_uid:
                user2_bot1_session = session
                break
        
        assert user2_bot1_session is not None
        assert user2_bot1_session["bot_id"] == bot1_id
        assert user2_bot1_session["firebase_uid"] == user2.firebase_uid
        assert "user" in user2_bot1_session
        assert user2_bot1_session["user"]["firebase_uid"] == user2.firebase_uid
        
        # Test 3: User1 should NOT see any sessions for bot3 (owned by user2)
        user1_sessions_for_bot3 = db.fetch_sessions_by(bot_id=bot3_id, firebase_uid=None, user=user1)
        # User1 should only see their own session with bot3, not user2's
        assert len(user1_sessions_for_bot3) == 1
        assert user1_sessions_for_bot3[0]["firebase_uid"] == user1.firebase_uid
        assert user1_sessions_for_bot3[0]["bot_id"] == bot3_id
        
        # Test 4: User2 should see all sessions for bot3 (which they own)
        user2_sessions_for_bot3 = db.fetch_sessions_by(bot_id=bot3_id, firebase_uid=None, user=user2)
        assert len(user2_sessions_for_bot3) == 2  # Should see both their own and user1's sessions
        
        # Check that both user1 and user2 sessions are present
        user1_found = False
        user2_found = False
        for session in user2_sessions_for_bot3:
            if session["firebase_uid"] == user1.firebase_uid:
                user1_found = True
            if session["firebase_uid"] == user2.firebase_uid:
                user2_found = True
        
        assert user1_found, "User2 should see user1's session with bot3"
        assert user2_found, "User2 should see their own session with bot3"
        
        # Test 5: User2 should NOT see user1's session for bot1 (owned by user1)
        user2_sessions_for_bot1 = db.fetch_sessions_by(bot_id=bot1_id, firebase_uid=None, user=user2)
        assert len(user2_sessions_for_bot1) == 1  # Should only see their own session
        assert user2_sessions_for_bot1[0]["firebase_uid"] == user2.firebase_uid
        
        # Test 6: When no bot_id is specified, users should only see their own sessions
        user1_all_sessions = db.fetch_sessions_by(bot_id=None, firebase_uid=user1.firebase_uid, user=user1)
        assert len(user1_all_sessions) == 2  # Should see their sessions with bot1 and bot3
        
        user1_sessions_count = 0
        for session in user1_all_sessions:
            if session["firebase_uid"] == user1.firebase_uid:
                user1_sessions_count += 1
        
        assert user1_sessions_count == 2, "User1 should see exactly their 2 sessions"
        
        # Test 7: Test bot deletion functionality
        
        # Test 7.1: User cannot delete a non-existent bot
        non_existent_bot_id = f"test_bot_{uuid.uuid4()}"
        assert db.delete_bot(non_existent_bot_id, user1) is False, "Should not be able to delete non-existent bot"
        
        # Test 7.2: User cannot delete another user's bot
        assert db.delete_bot(bot3_id, user1) is False, "User1 should not be able to delete User2's bot"
        # Verify bot3 still exists
        assert db.load_bot(bot3_id) is not None, "Bot3 should still exist after failed deletion attempt"
        
        # Test 7.3: User can delete their own bot
        assert db.delete_bot(bot2_id, user1) is True, "User1 should be able to delete their own bot"
        # Verify bot2 no longer exists
        assert db.load_bot(bot2_id) is None, "Bot2 should no longer exist after deletion"
        
        # Test 7.4: Verify bot is removed from browse results after deletion
        user1_bots_after_delete = db.browse_bots(user1)
        assert bot1_id in user1_bots_after_delete, "Bot1 should still be in browse results"
        assert bot2_id not in user1_bots_after_delete, "Bot2 should not be in browse results after deletion"
        
    finally:
        
        # Clean up test data
        db.db.collection(db.BOT_COLLECTION + db.DB_VERSION).document(bot1_id).delete()
        # bot2_id already deleted in test
        db.db.collection(db.BOT_COLLECTION + db.DB_VERSION).document(bot3_id).delete()
        
        # Clean up test sessions
        db.db.collection(db.CONVERSATION_COLLECTION + db.DB_VERSION).document(session1_id).delete()
        db.db.collection(db.CONVERSATION_COLLECTION + db.DB_VERSION).document(session2_id).delete()
        db.db.collection(db.CONVERSATION_COLLECTION + db.DB_VERSION).document(session3_id).delete()
        db.db.collection(db.CONVERSATION_COLLECTION + db.DB_VERSION).document(session4_id).delete()