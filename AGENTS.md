# Repository Guidelines

LightRAG is an advanced Retrieval-Augmented Generation (RAG) framework designed to enhance information retrieval and generation through graph-based knowledge representation.

## Project Structure & Module Organization
- `lightrag/`: Core Python package with orchestrators (`lightrag/lightrag.py`), storage adapters in `kg/`, LLM bindings in `llm/`, and helpers such as `operate.py` and `utils_*.py`.
- `lightrag-api/`: FastAPI service (`lightrag_server.py`) with routers under `routers/` and Gunicorn launcher `run_with_gunicorn.py`.
- `lightrag_webui/`: React 19 + TypeScript client driven by Bun + Vite; UI components live in `src/`.
- Tests live in `tests/` and root-level `test_*.py`. Working datasets stay in `inputs/`, `rag_storage/`, `temp/`; deployment collateral lives in `docs/`, `k8s-deploy/`, and `docker-compose.yml`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: set up the Python runtime.
- `pip install -e .` / `pip install -e .[api]`: install the package and API extras in editable mode.
- `lightrag-server` or `uvicorn lightrag.api.lightrag_server:app --reload`: start the API locally; ensure `.env` is present.
- `python -m pytest tests` (offline markers apply by default) or `python -m pytest tests --run-integration` / `python test_graph_storage.py`: run the full suite, opt into integration coverage, or target an individual script.
- `ruff check .`: lint Python sources before committing.
- `bun install`, `bun run dev`, `bun run build`, `bun test`: manage the web UI workflow (Bun is mandatory).

## Coding Style & Naming Conventions
- Backend code follow PEP 8 with four-space indentation, annotate functions, and reach for dataclasses when modelling state.
- Use `lightrag.utils.logger` instead of `print`; respect logger configuration flags.
- Extend storage or pipeline abstractions via `lightrag.base` and keep reusable helpers in the existing `utils_*.py`.
- Python modules remain lowercase with underscores; React components use `PascalCase.tsx` and hooks-first patterns.
- Front-end code should remain in TypeScript with two-space indentation, rely on functional React components with hooks, and follow Tailwind utility style.

## Testing Guidelines
- Keep pytest additions close to the code you touch (`tests/` mirrors feature folders and there are root-level `test_*.py` helpers); functions must start with `test_`.
- Follow `tests/pytest.ini`: markers include `offline`, `integration`, `requires_db`, and `requires_api`, and the suite runs with `-m "not integration"` by default—pass `--run-integration` (or set `LIGHTRAG_RUN_INTEGRATION=true`) when external services are available.
- Use the custom CLI toggles from `tests/conftest.py`: `--keep-artifacts`/`LIGHTRAG_KEEP_ARTIFACTS=true`, `--stress-test`/`LIGHTRAG_STRESS_TEST=true`, and `--test-workers N`/`LIGHTRAG_TEST_WORKERS` to dial up workloads or preserve temp files during investigations.
- Export other required `LIGHTRAG_*` environment variables before running integration or storage tests so adapters can reach configured backends.
- For UI updates, pair changes with Vitest specs and run `bun test`.

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects (e.g., `Fix lock key normalization`) and add body context only when necessary.
- PRs should include a summary, operational impact, linked issues, and screenshots or API samples for user-facing work.
- Verify `ruff check .`, `python -m pytest`, and affected Bun commands succeed before requesting review; note the runs in the PR text.

## Security & Configuration Tips
- Copy `.env.example` and `config.ini.example`; never commit secrets or real connection strings.
- Configure storage backends through `LIGHTRAG_*` variables and validate them with `docker-compose` services when needed.
- Treat `lightrag.log*` as local artefacts; purge sensitive information before sharing logs or outputs.

## Automation & Agent Workflow
- Use repo-relative `workdir` arguments for every shell command and prefer `rg`/`rg --files` for searches since they are faster under the CLI harness.
- Default edits to ASCII, rely on `apply_patch` for single-file changes, and only add concise comments that aid comprehension of complex logic.
- Honor existing local modifications; never revert or discard user changes (especially via `git reset --hard`) unless explicitly asked.
- Follow the planning tool guidance: skip it for trivial fixes, but provide multi-step plans for non-trivial work and keep the plan updated as steps progress.
- Validate changes by running the relevant `ruff`/`pytest`/`bun test` commands whenever feasible, and describe any unrun checks with follow-up guidance.


刚开始第一个段子，一定要能够吸引人的眼球，不然别人都不愿意驻足留下来看你的视频，也压根不知道你是在讲新概念成语了。
所以我们前置的段子，应该选择那些有话题争议性，能够引起男女辩论的东西最好。而且要贴近生活，让人能够有共鸣。而且不能太空洞，需要生活化一点的段子。
然后我现在想的是：袜子和内衣内裤能不能放洗衣机一起洗  从这一个切入点着手
因为“袜子能不能和内裤一起洗”是互联网永恒的辩论题。很多情侣都会因为洗衣服吵架，而且男生偷懒不想分类这个点，共鸣度简直爆炸。
然后针对之前说的 煞有介事 这个词，这个场景能很方便的加入男生的胡扯狡辩，和这个词吻合，能方便的引出那句话。

场景： 卫生间。你抱着一堆衣服，正打算用洗衣机洗衣服。结果打开洗衣机发现，里面有我正在洗的内裤、袜子。我又没有按照要求将他们分开洗。我正躺在沙发上玩游戏，你马上生气的大叫一声，找我理论。眼神犀利。

0-5秒（冲突爆发）： 你：“停！！你又要干嘛？这袜子能跟内衣一起洗吗？我都说了八百遍了，细菌会交叉感染！” 
我：（没有丝毫慌张，反而轻蔑一笑，推了推并不存在的眼镜）“交叉感染？宝贝，你对现代洗涤动力学的理解，还停留在农耕时代。”

5-15秒（第一波专家输出——物理学忽悠）： 你：“……洗个袜子跟农耕有什么关系？” 
我：（指着洗衣机内筒，语速极快）“听着，洗衣机高速旋转时会产生 800 转的‘离心风暴’。在这个转速下，袜子上的细菌会因为‘晕车’而失去附着力！这时候，洗衣液里的活性酶会趁虚而入，把它们全部瓦解！如果分开洗，水流不够猛烈，细菌反而会在温水里泡温泉！只有混洗，才能利用‘袜子的粗糙面料’去摩擦内衣，达到物理抛光的效果！” 她：“……物理抛光？那真菌呢？”

15-25秒（第二波专家输出——生物学忽悠）： 我：（立刻打断，语气更加严肃）“问得好！这就涉及到了‘微生物社会学’。你以为分开洗就干净了？错！那是温室里的花朵！我们把袜子放进去，实际上是在进行一种‘可控的抗原入侵’。这会让你的内衣产生危机感，从而激发棉纤维的‘应激张力’！这种洗出来的衣服穿在身上，才能真正帮你的皮肤建立起一道‘全天候生物防火墙’！这叫什么？这叫‘狼性洗涤法’！” 
你：（被你的连珠炮轰得有点懵）“……但我不想穿狼性的内衣啊。” 
我：（把袜子狠狠扔进去，一锤定音）“那是你的格局还没打开！这是在为你的免疫系统赋能！”

5-12秒（煞有介事的胡扯）： 她：“……洗个衣服还要溺爱？” 
我：（严肃地拿起一只袜子和一件内衣）“衣服也是有社交圈的！你长期把它们隔离，会让它们产生‘面料洁癖’！只有把袜子（细菌多）和内衣（干净）放在一起进行‘菌群交换大乱斗’，才能锻炼出最强的纤维抵抗力！这叫‘以毒攻毒’！洗出来的衣服穿在身上才百毒不侵！”
 你：（听傻了）“你不想手洗袜子直说……” 
 我：“我这是在帮它们建立‘群体免疫’！”

25-30秒（收尾）： 你（终于回过神，把袜子捞出来甩你脸上）：“你明明就是懒得手洗，你明明在胡扯，还装得像专家一样认真。” 
我：“胡扯还像专家一样认真——煞有介事。”。” 