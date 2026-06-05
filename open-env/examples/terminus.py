from terminus_env import TerminusEnv

with TerminusEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(
        setup=["mkdir -p /home/user/work"],
        verify=["test -f /home/user/work/answer.txt"],
    )
    print(env.call_tool("terminal", command="echo done > /home/user/work/answer.txt"))
    print(env.call_tool("terminal", final_answer="done"))