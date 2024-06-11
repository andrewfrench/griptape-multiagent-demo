from duckduckgo_search import DDGS
from dotenv import load_dotenv

from griptape.artifacts import TextArtifact 
from griptape.drivers import LocalStructureRunDriver 
from griptape.rules import Rule 
from griptape.structures import Agent, Pipeline, Workflow
from griptape.tasks import CodeExecutionTask, PromptTask, StructureRunTask
from griptape.tools import StructureRunClient, TaskMemoryClient, WebScraper, FileManager

from util import kebab

load_dotenv()



def search_ddg(task: CodeExecutionTask) -> TextArtifact:
    keywords = task.input.value
    results = DDGS().text(keywords, max_results=4)
    return TextArtifact(str(results))


def build_search_pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_task(
            CodeExecutionTask(
                "{{ args[0] }}",
                run_fn=search_ddg,
            ),
        )

    return pipeline


search_driver = LocalStructureRunDriver(structure_factory_fn=build_search_pipeline)
search_tool = StructureRunClient(
    name="Search Tool",
    description="Search the Web for information",
    driver=search_driver,
    off_prompt=True,
)


def build_researcher():
    return Agent(
        id="researcher",
        tools=[
            search_tool,
            WebScraper(off_prompt=True),
            TaskMemoryClient(off_prompt=False)
        ],
        rules=[
            Rule("Role: Lead Research Analyst"),
            Rule("Objective: Discover innovative advancements in artificial intelligence and data analytics"),
            Rule("Background: You are part of a prominent technology research institute"),
            Rule("Use the WebScraper to gather additional information from the URLs returned by the web search")
        ],
    )


def build_writer_fn(role: str, objective: str, style: str):
    def returns_writer():
        return Agent(
            id=f"{kebab(role)}-writer",
            tools=[
                TaskMemoryClient(off_prompt=False),
                FileManager(),
            ],
            rules=[
                Rule(f"Role: {role}"),
                Rule(f"Objective: {objective}"),
                Rule(f"Your writing style: {style}"),
                Rule("Prefer paragraphs over bulleted or numbered lists"),
                Rule(f"Use the FileManager to save your article as {kebab(role)}.md"),
                Rule("Use the provided insights to create a blog post in your style and tailored to your role")
            ],
        )

    return returns_writer


team = Workflow()

research_task = team.add_task(
    StructureRunTask(
        ("Perform a detailed examination of the newest developments in AI as of 2024. "
         "Pinpoint major trends, breakthroughs, and their implications for various industries.",),
        id="research",
        driver=LocalStructureRunDriver(
            structure_factory_fn=build_researcher,
        ),
    ),
)

writers = [
    {
        "role": "The AI Optimist",
        "goal": "Convince everyone that AI will solve all their problems.",
        "style": "Writes at a college level. Breathless excitement for AI advancements, prone to exaggeration. "
                 "Focused entirely on the positive disruptions AI advancements may bring, without any "
                 "consideration of the risks."
    },
    {
        "role": "The AI Pessimist",
        "goal": "Convince everyone that AI will ruin everything.",
        "style": "Writes at the level of a academic journal. Cynical and negative, terse, and unexcited. Focused "
                 "entirely on the negative disruptions and risks of AI advancements, without any consideration "
                 "for the benefits."
    },
    {
        "role": "The Fool",
        "goal": "Make everyone think you're smart by using big words and phrases",
        "style": "Writes at a third-grade level. Grammatical errors, run-on sentences, and thoughts that lead nowhere "
                 "abound. Clearly confused, with spelling mistakes and misunderstandings throughout, demonstrating a "
                 "clear lack of relevant knowledge."
    }
]

writing_agent_tasks = []
for writer in writers:
    writing_agent_tasks.append(
        StructureRunTask(
            ("""
                Using the insights provided, develop a blog post in the style requested.

                Insights:
                {{ parent_outputs["research"] }}
                """,),
            driver=LocalStructureRunDriver(
                structure_factory_fn=build_writer_fn(
                    role=writer["role"],
                    objective=writer["goal"],
                    style=writer["style"],
                )
            ),
            id=f"writer-agent-task-{kebab(writer['role'])}"
        )
    )


end_task = team.add_task(
    CodeExecutionTask(
        run_fn=lambda _: TextArtifact("All done!")
    )
)

team.insert_tasks(research_task, writing_agent_tasks, end_task)

if __name__ == "__main__":
    team.run()
