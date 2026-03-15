from fastapi import WebSocket, WebSocketDisconnect
from dishka.integrations.fastapi import FromDishka
from agents.coder import coder_agent, AgentDeps
from services.docker_sandbox import DockerSandboxManager
from services.code_search import CodeSearchService
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
from services.indexing import IndexingService
import json

class AgentSocketController:
    """
    Controller for handling agent WebSocket sessions.
    """
    docker: DockerSandboxManager
    search: CodeSearchService
    editor: FileEditorService
    git: GitManagerService
    indexing: IndexingService

    def __init__(
        self,
        docker: DockerSandboxManager,
        search: CodeSearchService,
        editor: FileEditorService,
        git: GitManagerService,
        indexing: IndexingService,
    ):
        self.docker = docker
        self.search = search
        self.editor = editor
        self.git = git
        self.indexing = indexing

    async def handle_session(
        self,
        websocket: WebSocket,
    ) -> None:
        """
        Handle a WebSocket session for real-time agent interaction.
        """
        await websocket.accept()
        
        deps = AgentDeps(
            docker=self.docker,
            search=self.search,
            editor=self.editor,
            git=self.git
        )
        
        try:
            # Wait for the first message which should contain the prompt and config
            data = await websocket.receive_text()
            message_data = json.loads(data)
            initial_prompt = message_data.get('prompt', '')
            coder_model = message_data.get('coder_model', 'ollama:llama3')
            reviewer_model = message_data.get('reviewer_model', 'ollama:llama3')
            embedding_model = message_data.get('embedding_model', 'llama3')
            repo_path = message_data.get('repo_path')

            if repo_path:
                import os
                os.environ['REPO_PATH'] = repo_path
                os.environ['EMBEDDING_MODEL'] = embedding_model
                # Re-initialize services with the new path
                self.docker.repo_path = repo_path
                self.editor.root_path = repo_path
                self.git.repo_path = repo_path
                self.indexing.repo_path = repo_path

            if not initial_prompt:
                await websocket.send_json({'type': 'error', 'content': 'No prompt provided.'})
                await websocket.close()
                return
            
            await websocket.send_json({'type': 'thought', 'content': 'Indexing repository...'})
            await self.indexing.index_repository()
            
            await websocket.send_json({'type': 'thought', 'content': 'Initializing sandbox...'})
            await self.docker.spawn_sandbox()
            
            # Use a unique branch name per session
            import uuid
            session_id = str(uuid.uuid4())[:8]
            branch_name = f'ai-task-{session_id}'
            await self.git.create_branch(branch_name)
            
            await websocket.send_json({'type': 'thought', 'content': 'Generating plan...'})
            
            # Generate actual plan using the agent
            plan_prompt = f'Task: {initial_prompt}\n\nPlease generate a step-by-step plan. Do not execute any tools yet.'
            plan_result = await coder_agent.run(plan_prompt, deps=deps, model=coder_model)
            
            await websocket.send_json({
                'type': 'plan',
                'content': plan_result.data, # type: ignore
                'requires_approval': True
            })
            
            # Wait for approval
            approval_data = await websocket.receive_text()
            if json.loads(approval_data).get('approved'):
                await websocket.send_json({'type': 'thought', 'content': 'Executing plan...'})
                
                from agents.reviewer import reviewer_agent
                
                # Execute agent run with streaming
                execution_prompt = f'Task: {initial_prompt}\n\nPlan: {plan_result.data}\n\nPlease execute the plan.' # type: ignore
                async with coder_agent.run_stream(execution_prompt, deps=deps, model=coder_model) as result:
                    async for event in result.stream_responses():
                        # Stream thoughts (messages from model)
                        await websocket.send_json({'type': 'thought', 'content': str(event)})
                    
                    final_result: str = await result.get_data() # type: ignore
                    await websocket.send_json({'type': 'thought', 'content': 'Coder finished. Starting review...'})

                    # Review phase
                    diff = await self.git.get_diff()
                    review = await reviewer_agent.run(f'Please review these changes:\n\n{diff}', model=reviewer_model)
                    
                    review_data = review.data # type: ignore
                    await websocket.send_json({
                        'type': 'review',
                        'content': review_data.comments,
                        'approved': review_data.approved
                    })

                    if review_data.approved:
                        await websocket.send_json({'type': 'done', 'content': final_result})
                    else:
                        await websocket.send_json({'type': 'error', 'content': 'Review failed. Feedback: ' + review_data.comments})
            else:
                await websocket.send_json({'type': 'error', 'content': 'Plan rejected by user.'})
                
        except WebSocketDisconnect:
            await self.docker.stop_sandbox()
        except Exception as e:
            await websocket.send_json({'type': 'error', 'content': str(e)})
            await websocket.close()
