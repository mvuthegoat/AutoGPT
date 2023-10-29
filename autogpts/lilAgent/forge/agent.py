from forge.sdk import (
    Agent,
    AgentDB,
    ForgeLogger,
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    Workspace,    
    PromptEngine,	
    chat_completion_request,	
    ChromaMemStore	
)
import json	
import pprint

import os

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code, so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally, based on the profile selected, the agent could be configured to use a
    different llm. The possibilities are endless and the profile can be selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to accumulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensing short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agent's decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        For a tutorial on how to add your own logic please see the offical tutorial series:
        https://aiedge.medium.com/autogpt-forge-e3de53cc58ec

        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the benchmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentially the same as the task request and contains an input
        string, for the benchmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """

        task = await self.db.get_task(task_id)

        # An example that
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=True
        )
        

        # Set up working dir and store files
        self.wd = os.path.join(str(self.workspace.base_path), task.task_id)
        LOG.info(f"💽 Set working directory: {self.wd}")
        if not os.path.exists(self.wd):
            os.makedirs(self.wd)
        os.chdir(self.wd)

        self.files = {f:os.path.getmtime(os.path.join(self.wd, f)) for f in os.listdir(self.wd) if os.path.isfile(os.path.join(self.wd, f))}
        
        if self.files.keys():
            LOG.info("📂 Current files")
            LOG.info(str(self.files.keys()))


        LOG.info(f"Step_request: {step_request}")
        # Log the message
        #LOG.info(f"\t✅ Final Step completed: {step.step_id} input: {step.input[:19]}")
        
        # Initialize the PromptEngine with the "gpt-3.5-turbo" model
        prompt_engine = PromptEngine("gpt-3.5-turbo")
        # Load the system and task prompts
        system_prompt = prompt_engine.load_prompt("system-format")

        LOG.info(f"SYS_PROMPT: {system_prompt}")

        # Initialize the messages list with the system prompt
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Define the task parameters
        task_kwargs = {
            "task": task.input,
            "abilities": self.abilities.list_abilities_for_prompt(),
        }

        # Load the task prompt with the defined task parameters
        task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)

        #LOG.info(f"TASK_PROMPT: {task_prompt}")

        # Append the task prompt to the messages list
        messages.append({"role": "user", "content": task_prompt})

        count = 0
        
        while True:
            if count >= 10: 
                break
            count += 1

            success = False
            MAX_ATTEMPTS  = 3

            for attempt in range(MAX_ATTEMPTS):
                try:
                    LOG.info(f"TASK_PROMPT: {messages[-1]['content']}")
                    # Define the parameters for the chat completion request
                    chat_completion_kwargs = {
                        "messages": messages,
                        "model": "gpt-4",
                    }
                    # Make the chat completion request and parse the response
                    chat_response = await chat_completion_request(**chat_completion_kwargs)
                    answer = json.loads(chat_response["choices"][0]["message"]["content"])

                    # Log the answer for debugging purposes
                    LOG.info(pprint.pformat(answer))
                    success = True
                    break

                except json.JSONDecodeError as e:
                    # Handle JSON decoding errors
                    LOG.error(f"Unable to decode chat response: {chat_response}")
                except Exception as e:
                    # Handle other exceptions
                    LOG.error(f"Unable to generate chat response: {e}")

            if not success:
                LOG.error(f"Model can't generate good responses")
                break

            # Extract the ability from the answer
            ability = answer["ability"]

            # Run the ability and get the output
            if ability["name"] in self.abilities.list_abilities().keys():
                output = await self.abilities.run_ability(
                    task_id, ability["name"], **ability["args"]
                )
                LOG.info(f"Run ability output: {(output)}")
            elif ability["name"] != "inherent knowledge":
                ability_json = {
                    "ability": {
                        "name": "inherent knowledge",
                        "args": {
                            "result": "replace this with the result obtained from using your own inherent capabilities to solve the task",
                        }
                    }
                }
                ability_json = json.dumps(ability_json, indent=4)

                output = f"""The ability you used does not exist in the database. 
                But since you are a powerful LLM, you are equipped with that ability. 
                Now, please execute that ability to get the result. The ability keyword in your answer must now becomes:\n{ability_json}"""
                
                messages.append({"role": "user", "content": output})
                continue
            else:
                output = ability["args"]["result"]

            if output == None or ability["name"] == "finish":
                # Set the step output to the "speak" part of the answer
                step.output = answer["thoughts"]["speak"]
                break
            else:
                new_message = f'Here is the result after completing the previous step of the plan and {ability["name"]} ability\n' + str(output) + "\nNow, execute the next step of the plan"

                # Append the new_messages to the messages list
                messages.append({"role": "user", "content": new_message})

            # Set the step output to the "speak" part of the answer
            step.output = answer["thoughts"]["speak"]


        
        for file_path in os.listdir(os.getcwd()):
            # skip directories
            if os.path.isdir(os.path.join(self.wd, file_path)):
                continue
            if self.files.get(file_path, "") != "" or self.files.get(file_path, 0) != os.path.getmtime(os.path.join(self.wd, file_path)):
                LOG.info(f"🗒️ Create artifact {file_path}")

                await self.db.create_artifact(
                    task_id=task_id,
                    step_id=step.step_id,
                    file_name=file_path,
                    relative_path="",
                    agent_created=True,
                )
                self.files[file_path] = os.path.getmtime(os.path.join(self.wd, file_path))

        LOG.info(f"\t✅ Final Step completed: {step.step_id} output: {step.output}")
        return step
