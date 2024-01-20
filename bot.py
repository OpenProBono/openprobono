# OPB bot main function
def opb_bot(history, tools, user_prompt, session):
    t1name = 'government-search'
    t1txt = 'site:*.gov | site:*.edu | site:*scholar.google.com'
    t1prompt = 'Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.'
    t2name = 'case-search'
    t2txt = 'site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com'
    t2prompt = 'Use for finding case law. Always cite your sources.'
    user_prompt = ''

    if(history[-1][0].strip() == ""):
        history[-1][1] = "Hi, how can I assist you today?"
        yield history 
    else:
        q = Queue()
        job_done = object()

        history_langchain_format = ChatMessageHistory()
        for i in range(1, len(history)-1):
            (human, ai) = history[i]
            if human:
                history_langchain_format.add_user_message(human)
            if ai:
                history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")
        ##----------------------- tools -----------------------##
        #Helper function for concurrent processing of search results, calls the summarizer llm
        def search_helper_summarizer(result):
            result.pop("displayed_link", None)
            result.pop("favicon", None)
            result.pop("about_page_link", None)
            result.pop("about_page_serpapi_link", None)
            result.pop("cached_page_link", None)
            result.pop("snippet_highlighted_words", None)

            summary_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-16k-0613')
            llm_input = """Summarize this web page in less than 50 words.

            Web Page:
            """
            llm_input += str(UnstructuredURLLoader(urls=[result["link"]]).load())[:16385]
            result["page_summary"] = summary_llm.predict(llm_input)
            return result

        #Filter search results retured by serpapi to only include relavant results
        def process_search(results):
            new_dict = {}
            # if('sports_results' in results):
            #     new_dict['sports_results'] = results['sports_results']
            if('organic_results' in results):
                new_dict['organic_results'] = [search_helper_summarizer(result) for result in results['organic_results']]

            return new_dict
        
        toolset = [
        Tool(
        name = t["name"], 
        func = def search_tool(qr):
                data = {"search": t['txt'] + " " + qr, 'prompt': t['prompt'], 'timestamp': firestore.SERVER_TIMESTAMP}
                db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
                return process_search(GoogleSearch({
                    'q': t['txt'] + " " + qr,
                    'num': 5
                    }).get_dict()),
        coroutine = async def async_search_tool(qr):
            return search_tool(qr),
        description = t1['description']
        ) for tool in tools]
        tool_names = [tool.name for tool in toolset]

        ##----------------------- end of tools -----------------------##


        #------- agent definition -------#
        # Set up the base template
        template = user_prompt + """Respond the user as best you can. You have access to the following tools:

        {tools}

        The following is the chat history so far:
        {memory}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question, including your sources.

        These were previous tasks you completed:



        Begin!

        {input}
        {agent_scratchpad}"""

        # Set up a prompt template
        class CustomPromptTemplate(BaseChatPromptTemplate):
            # The template to use
            template: str
            # The list of tools available
            tools: List[Tool]

            def format_messages(self, **kwargs) -> str:
                # Get the intermediate steps (AgentAction, Observation tuples)
                # Format them in a particular way
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                # Set the agent_scratchpad variable to that value
                kwargs["agent_scratchpad"] = thoughts
                # Create a tools variable from the list of tools provided
                kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
                # Create a list of tool names for the tools provided
                kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
                formatted = self.template.format(**kwargs)
                return [HumanMessage(content=formatted)]
            
        class CustomOutputParser(AgentOutputParser):
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                print(llm_output)
                print('inside parse')
                llm_output = '\n' + llm_output
                q.put(llm_output)
                # Check if agent should finish
                if "Final Answer:" in llm_output:
                    print('inside final answer')
                    # q.put(llm_output.split("Final Answer:")[-1])
                    return AgentFinish(
                        # Return values is generally always a dictionary with a single `output` key
                        # It is not recommended to try anything else at the moment :)
                        return_values={"output": llm_output.split("Final Answer:")[-1]},
                        log=llm_output,
                    )
                # Parse out the action and action input
                regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    print('inside no match')
                    # q.put(llm_output) #.split("Question:")[-1].split("\n")[0])
                    # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                    return AgentFinish(
                        # Return values is generally always a dictionary with a single `output` key
                        # It is not recommended to try anything else at the moment :)
                        return_values={"output": llm_output}, #.split("Question:")[-1].split("\n")[0]},
                        log=llm_output,
                    )
                action = match.group(1).strip()
                action_input = match.group(2)
                # Return the action and action input
                # q.put("Processing...\n")
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        prompt_template = CustomPromptTemplate(
            template=template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps", "memory"]
        )

        output_parser = CustomOutputParser()
        #------- end of agent definition -------#
        async def task(prompt):
            #definition of llm used for bot
            bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613', request_timeout=60*5)
            agent_kwargs = {
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            }
            llm_chain = LLMChain(llm=bot_llm, prompt=prompt_template)
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)
            ret = await agent_executor.arun(prompt)
            q.put(job_done)
            return ret

        with start_blocking_portal() as portal:
            portal.start_task_soon(task, history[-1][0])

            content = ""
            while True:
                next_token = q.get(True)
                if next_token is job_done:
                    break
                content += next_token
                history[-1] = (history[-1][0], content)

                yield history



#TODO: cache vector db with bot_id
#TODO: do actual chat memory
#TODO: try cutting off intro and outro part of videos
def youtube_bot(
    history,
    bot_id,
    user_prompt = "",
    youtube_urls = [],
    session = ""):

    if(user_prompt is None or user_prompt == ""):
        user_prompt = "Respond in the same style as the youtuber in the context below."

    prompt_template = user_prompt + """
    \n\nContext: {context}
    \n\n\n\n
    Question: {question}
    Response:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    embeddings = OpenAIEmbeddings()
    bot_path = "./youtube_bots/" + bot_id
    try:
        vectordb = FAISS.load_local(bot_path, embeddings)
    except:
        text = ""
        for url in youtube_urls:
            try:
                # Load the audio
                loader = YoutubeLoader.from_youtube_url(
                    url, add_video_info=False
                )
                docs = loader.load()
                # Combine doc
                combined_docs = [doc.page_content for doc in docs]
                text += " ".join(combined_docs)
            except:
                print("Error occured while loading transcript from video with url: " + url)

        # Split them
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_text(text)

        # Build an index
        vectordb = FAISS.from_texts(splits, embeddings)
        vectordb.save_local(bot_path)

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )

    query = history[-1][0]
    #check for empty query
    if(query.strip() == ""):
        history[-1][1] = ""
    else:
        history[-1][1] = qa_chain.run(query)

    return history