# GitHub 同步与 AI Git 规范

本项目使用 GitHub 私有仓库同步代码。本地 Windows 是主要开发端，Linux GPU 服务器是
实验执行端。Git 保存代码和公开实验配置；模型、环境、原始结果与登录资料独立保存。
本文件约束 AI 和人工操作，用户当次明确指令优先。

当前远程仓库为 `QingHuan-6/InferenceOptim`（Private），本地目录名仍为 AInferenceOptim。
origin 使用 `https://github.com/QingHuan-6/InferenceOptim.git`；服务器可使用对应 SSH 地址。

## 分支与同步

- `main` 保存已通过检查、可部署的版本；日常修改放在短期工作分支。
- 工作分支统一命名为 `ai/<type>-<topic>`，例如 `ai/feat-adaptive-draft`、
  `ai/fix-eos-handling`。topic 使用英文小写和连字符。
- 不使用长期 develop 分支，不为每次小修补增加多层分支。
- 初次建仓库可在 main 创建一个经过检查的基础提交，此后使用工作分支。
- 本地提交后推送对应分支，通过 PR 将完整功能合入 main；默认使用 squash 合并。
- 用户明确允许直接更新 main 时可以执行，但仍须完成检查并使用普通快进推送。
- GPU 服务器默认只拉取 main；需要测试工作分支时明确记录分支和 commit。

## AI 操作边界

1. 开始修改前读取 AGENTS.md、工程规范和本文件，检查 `git status --short --branch`、
   `git diff`、当前分支与 `git remote -v`。尚未初始化时先完成本地初始化。
2. 保护已有修改：不丢弃、不覆盖、不擅自提交用户或其他任务的改动。
   新任务遇到既有修改时，只处理能明确归属本任务的文件或代码块；无法区分时先澄清。
3. 对已授权开发任务，可建立工作分支，完成实现、检查、审阅差异后创建本地提交。
   用户要求仅修改、不提交时遵从。未完成或检查失败的工作不能包装成已验证提交。
4. 推送 GitHub、创建远程仓库、PR 或合并，需要当次请求或先前明确授权覆盖该操作。
   用户要求“同步到 GitHub”即授权向约定仓库推送当前任务工作分支，无需重复确认。
   未指定仓库或远程分支时，先核对；不猜测账号、仓库或公开范围。
5. 未明确要求公开时使用私有仓库；不得擅自变更可见性、权限或分支保护。
6. 禁止自行执行 force push、reset --hard、clean -fd、删除分支、改写共享提交或
   覆盖冲突文件。已有提交需要修正时默认新增提交；回滚共享变更使用 revert。
7. 禁止伪造测试结果、用户身份、作者、签名和远程执行结果，不修改全局 Git 配置。
   没有可用提交身份时说明缺失项，由用户提供；不要用虚构邮箱补齐。
8. Git 凭据使用用户自己的 SSH key 或凭据管理器；不写入仓库 URL、脚本或文档，
   不要求用户在聊天中粘贴 token、私钥或密码。

## 提交规范

提交格式：`type(scope): 中文说明`，例如：

```text
feat(backends): 增加普通解码参考验证
fix(metrics): 修正单 token 输出的缺失值
test(backends): 覆盖 EOS 默认参数合并
docs(git): 明确服务器同步和提交规范
chore(repo): 初始化实验项目
```

type 使用 feat、fix、refactor、perf、test、docs、build、ci、chore。
scope 使用受影响的模块，例如 backends、config、metrics、runner、verify、deploy、repo。
标题尽量不超过 72 字符，不写“更新代码”“优化一下”等模糊描述。
perf 仅用于有测量依据的性能改进，不能仅凭代码形态宣称加速。

一次提交围绕一个目的，包含必要测试与文档，不混入无关格式化、依赖升级或临时实验。
正文按需解释问题、行为变化及验证限制；不虚构 GPU 验证或附带未执行的检查。

AI 暂存必须使用明确路径或 `git add -p`，禁止盲目 `git add .`、`git add -A`。
提交前审阅 `git diff --cached --stat` 和 `git diff --cached`，确认没有敏感资料。

## 检查与 PR

行为变化执行 unittest、Ruff lint、Ruff 格式检查与 mypy；Windows 可用
`./scripts/check.ps1`。仅文档修改无需反复运行模型集成测试，但须检查链接和示例命令，
并执行 `git diff --check`。安装了推理依赖时 CPU 集成测试应实际执行，不应跳过。
CUDA 实测单独注明设备、精度和结果位置，不能由 CPU 或 mock 测试替代。

PR 描述说明最终问题、行为、验证与限制。代码审阅后再合并；不在规范中要求绕过
用户已有的审阅、分支保护或自动检查。合并后记录 main 的最终 commit，
不要把 squash 前工作分支的 commit 当成最终 main 版本。

## 首次连接 GitHub

先在自己的 GitHub 下创建 **Private、空仓库**，不在网页预建 README 或 .gitignore。
在本地确认提交身份可用后，检查并仅暂存准备发布的源码、文档和配置，创建基础提交。
再将下面的地址替换为真实私有仓库地址：

```powershell
git remote add origin https://github.com/QingHuan-6/InferenceOptim.git
git push -u origin main
```

如果 origin 已存在，先检查，不直接覆盖。HTTPS 地址也可用，但认证交由凭据管理器。
创建空仓库和首次推送是不同步骤；本地初始化不表示 GitHub 已完成同步。

后续 AI 工作分支的示例：

```powershell
git switch main
git pull --ff-only
git switch -c ai/feat-adaptive-draft
# 开发、验证、明确路径暂存和提交后
git push -u origin ai/feat-adaptive-draft
```

示例中的 pull 只在 main 已有远程跟踪分支且工作区清洁时执行。
工作分支不直接以无参数 git pull 解决分叉；先 fetch、查看差异，再确定合并策略。

## 服务器同步

首次部署先配置私有仓库读取权限，建议使用该仓库只读 deploy key。
服务器不需要获得推送权限，不要复制本地完整凭据或 AutoDL 工具目录。

```bash
git clone git@github.com:QingHuan-6/InferenceOptim.git AInferenceOptim
cd AInferenceOptim
bash scripts/setup_4090.sh
```

每轮实验结束后再更新，先确保没有正在运行的实验或本地改动：

```bash
git status --short --branch
# 工作区清洁、处于 main 且没有实验运行时
git pull --ff-only
git rev-parse HEAD
```

pull --ff-only 失败时停止，不用 reset 或强推“修好”。依赖变化后按运行说明更新环境；
仅 Python 源码变化时，可编辑安装会直接使用新源码。
服务器上临时试验需要改配置时放到 gitignored 的 configs/local/，用明确路径调用，
避免修改已跟踪的预设而阻止后续 pull。有价值的修改回到本地整理后提交。

每轮结果至少关联代码 commit、工作区是否干净、输入快照、配置、模型 commit 和环境。
run 的 manifest 已自动采集 Git commit/dirty/status；Git 不可用时明确记录缺失。
verify/diagnose 尚需人工关联 Git 状态，不能仅记录分支名。
例如在工作区清洁时创建独立的版本记录，保存在输出目录，不能覆盖旧记录：

```bash
mkdir -p outputs
revision_file="outputs/code-revision-$(date -u +%Y%m%dT%H%M%S)-$$.txt"
(set -o noclobber; { git rev-parse HEAD; git status --porcelain=v1; } > "$revision_file")
```

源代码目录、模型缓存和实验输出必须分开；更新代码不自动清理后两者。
结果备份走独立下载或存储，不提交到 GitHub。

## 排除内容与行尾

不跟踪 .venv、.venv-4090、.local-tools、.runtime、模型、缓存、outputs、.env、
密钥和私有数据。configs/local/ 与 data/private/ 用于本机或服务器私有资料。
原始任务书暂保留本地，docs 下的 docx 默认排除；公开前另行明确决定。
添加忽略规则不能移除已经跟踪的文件，每次提交仍须审阅暂存区。

文本统一 LF、UTF-8，保证 Linux shell 脚本可直接使用；cmd/bat 使用 CRLF。
不设置自动部署、提交后自动租卡或自动重启实验。
