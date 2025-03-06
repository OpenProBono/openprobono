import os
import uuid

from app import db
from app.models import BotRequest, ChatModelParams, EngineEnum, OpenAIModelEnum, User, EvalDataset, EvalSession

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

def test_eval_dataset_operations() -> None:
    """Test storing and retrieving evaluation datasets."""
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
    
    # Generate unique IDs for the bots
    bot1_id = f"test_bot_{uuid.uuid4()}"
    bot2_id = f"test_bot_{uuid.uuid4()}"
    
    # Generate dataset IDs upfront
    dataset1_id = f"test_dataset_{uuid.uuid4()}"
    dataset2_id = f"test_dataset_{uuid.uuid4()}"
    dataset3_id = f"test_dataset_{uuid.uuid4()}"
    dataset_with_sessions_id = f"test_dataset_{uuid.uuid4()}"
    
    try:
        # Store the bots
        db.store_bot(bot1, bot1_id)
        db.store_bot(bot2, bot2_id)
        
        # Create test datasets
        dataset1 = EvalDataset(
            name="Test Dataset 1",
            description="A test dataset for user 1",
            inputs=["What is the capital of France?", "Explain quantum computing"],
            bot_ids=[bot1_id, bot2_id],
            user=user1
        )
        
        dataset2 = EvalDataset(
            name="Test Dataset 2",
            description="Another test dataset for user 1",
            inputs=["How does a car engine work?", "What is the theory of relativity?"],
            bot_ids=[bot1_id],
            user=user1
        )
        
        dataset3 = EvalDataset(
            name="Test Dataset 3",
            description="A test dataset for user 2",
            inputs=["What is machine learning?"],
            bot_ids=[bot1_id],
            user=user2
        )
        
        # Store the datasets
        db.store_eval_dataset(dataset1, dataset1_id)
        db.store_eval_dataset(dataset2, dataset2_id)
        db.store_eval_dataset(dataset3, dataset3_id)
        
        # Test retrieving datasets for user1
        user1_datasets = db.get_user_datasets(user1)
        assert dataset1_id in user1_datasets
        assert dataset2_id in user1_datasets
        assert dataset3_id not in user1_datasets
        
        # Test retrieving datasets for user2
        user2_datasets = db.get_user_datasets(user2)
        assert dataset1_id not in user2_datasets
        assert dataset2_id not in user2_datasets
        assert dataset3_id in user2_datasets
        
        # Test retrieving a specific dataset
        retrieved_dataset = db.get_dataset(dataset1_id)
        assert retrieved_dataset is not None
        assert retrieved_dataset.name == dataset1.name
        assert retrieved_dataset.description == dataset1.description
        assert retrieved_dataset.inputs == dataset1.inputs
        assert retrieved_dataset.bot_ids == dataset1.bot_ids
        assert retrieved_dataset.user.firebase_uid == user1.firebase_uid
        
        # Test sessions functionality
        # Create a dataset with sessions
        sessions = [
            EvalSession(
                input_idx=0,
                bot_idx=0,
                input_text="Question 1",
                output_text="Answer 1 from bot 1",
                bot_id=bot1_id,
                session_id="session1_1"
            ),
            EvalSession(
                input_idx=0,
                bot_idx=1,
                input_text="Question 1",
                output_text="Answer 1 from bot 2",
                bot_id=bot2_id,
                session_id="session1_2"
            ),
            EvalSession(
                input_idx=1,
                bot_idx=0,
                input_text="Question 2",
                output_text="Answer 2 from bot 1",
                bot_id=bot1_id,
                session_id="session2_1"
            ),
            EvalSession(
                input_idx=1,
                bot_idx=1,
                input_text="Question 2",
                output_text="Answer 2 from bot 2",
                bot_id=bot2_id,
                session_id="session2_2"
            )
        ]
        
        dataset_with_sessions = EvalDataset(
            name="Dataset with Sessions",
            description="A test dataset with sessions",
            inputs=["Question 1", "Question 2"],
            bot_ids=[bot1_id, bot2_id],
            sessions=sessions,
            user=user1
        )
        
        db.store_eval_dataset(dataset_with_sessions, dataset_with_sessions_id)
        
        # Retrieve and verify sessions
        retrieved_dataset_with_sessions = db.get_dataset(dataset_with_sessions_id)
        assert retrieved_dataset_with_sessions is not None
        assert len(retrieved_dataset_with_sessions.sessions) == 4
        
        # Check that all sessions are present
        session_ids = [s.session_id for s in retrieved_dataset_with_sessions.sessions]
        assert "session1_1" in session_ids
        assert "session1_2" in session_ids
        assert "session2_1" in session_ids
        assert "session2_2" in session_ids
        
        # Check that the input_idx and bot_idx are correct
        for session in retrieved_dataset_with_sessions.sessions:
            if session.session_id == "session1_1":
                assert session.input_idx == 0
                assert session.bot_idx == 0
                assert session.input_text == "Question 1"
                assert session.output_text == "Answer 1 from bot 1"
                assert session.bot_id == bot1_id
            elif session.session_id == "session1_2":
                assert session.input_idx == 0
                assert session.bot_idx == 1
                assert session.input_text == "Question 1"
                assert session.output_text == "Answer 1 from bot 2"
                assert session.bot_id == bot2_id
        
    finally:
        # Clean up test data
        db.db.collection(db.BOT_COLLECTION + db.DB_VERSION).document(bot1_id).delete()
        # db.db.collection(db.BOT_COLLECTION + db.DB_VERSION).document(bot2_id).delete()
        
        # db.db.collection(db.EVAL_DATASET_COLLECTION + db.DB_VERSION).document(dataset1_id).delete()
        # db.db.collection(db.EVAL_DATASET_COLLECTION + db.DB_VERSION).document(dataset2_id).delete()
        # db.db.collection(db.EVAL_DATASET_COLLECTION + db.DB_VERSION).document(dataset3_id).delete()
        # db.db.collection(db.EVAL_DATASET_COLLECTION + db.DB_VERSION).document(dataset_with_sessions_id).delete()