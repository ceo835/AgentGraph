"""Русскоязычный Streamlit-интерфейс для локального тестирования AgentGraph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Для запуска интерфейса установите optional extra `ui`: `pip install -e .[ui]`."
    ) from exc

from agentgraph import (
    build_demo_graph,
    build_demo_task,
    build_desktop_demo_graph,
    build_desktop_task,
    build_starter_agent_configs,
    describe_agent_configs,
    export_desktop_artifact,
    load_desktop_executor_config,
    resume_desktop_plan,
    resume_desktop_task,
    resume_task,
    run_desktop_plan,
    run_desktop_task,
    run_task,
)
from agentgraph.contracts import FactLogicValidationMode, HITLPoint, ThreadStatus

STATUS_LABELS = {
    ThreadStatus.INIT: "Подготовка",
    ThreadStatus.ROUTED: "Маршрут выбран",
    ThreadStatus.EXECUTING: "Выполняется",
    ThreadStatus.VALIDATING: "Проверяется результат",
    ThreadStatus.REPAIR: "Исправляется ответ",
    ThreadStatus.HITL_WAIT: "Ждёт вашего решения",
    ThreadStatus.MEMORY_SYNC: "Сохраняется контекст",
    ThreadStatus.COMPLETED: "Готово",
    ThreadStatus.FAILED: "Не удалось выполнить задачу",
}

AGENT_LABELS = {
    "coordinator": "Координатор",
    "researcher": "Исследователь",
    "synthesizer": "Синтезатор",
    "validator": "Валидатор",
    "planner": "Планировщик",
    "memory_curator": "Куратор памяти",
    "critic": "Критик",
    "file_executor": "Файловый исполнитель",
    "desktop_executor": "Desktop Assistant",
}

CHECKPOINT_REASON_LABELS = {
    "tool_approval": "нужно подтверждение перед выполнением действия",
    "before_tool_call": "нужно подтверждение перед выполнением действия",
    "before_delegation": "нужно подтверждение перед передачей задачи",
    "after_schema_validation_failure": "нужно решение после ошибки проверки структуры",
    "after_logic_validation_failure": "нужно решение после логической проверки",
    "on_low_confidence": "система недостаточно уверена в результате",
    "before_finalize": "нужно подтверждение перед завершением",
}

VALIDATION_MODE_OPTIONS = {
    "Без дополнительной проверки": FactLogicValidationMode.NONE,
    "Правила и бизнес-проверки": FactLogicValidationMode.POLICY,
    "Отдельный агент-валидатор": FactLogicValidationMode.SPECIALIST,
}

RUN_MODE_OPTIONS = {
    "Обычный агентный режим": "standard",
    "Desktop Assistant": "desktop",
}


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "graph": None,
        "run_result": None,
        "export_result": None,
        "events": [],
        "last_task": None,
        "last_error": None,
        "run_mode": "standard",
        "plan_execution": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _selected_hitl_points(confirm_before_action: bool) -> list[HITLPoint]:
    points: list[HITLPoint] = []
    if confirm_before_action:
        points.append(HITLPoint.BEFORE_TOOL_CALL)
    return points


def _build_agent_overview(include_critic: bool, run_mode: str) -> list[dict[str, Any]]:
    configs = build_starter_agent_configs(include_critic=include_critic)
    if run_mode == "desktop":
        configs = [*configs, load_desktop_executor_config()]
    return describe_agent_configs(configs)


def _should_use_desktop_flow(description: str, run_mode: str) -> bool:
    if run_mode == "desktop":
        return True
    desktop_task = build_desktop_task(query=description, locale="ru")
    return (
        desktop_task.assignee == "desktop_executor"
        or desktop_task.metadata.get("planner_domain") == "desktop"
    )


def _start_run(
    *,
    description: str,
    run_mode: str,
    env_file: str,
    use_planner: bool,
    validation_mode: FactLogicValidationMode,
    confirm_before_action: bool,
    include_critic: bool,
    use_stream: bool,
    allowed_paths_raw: str,
    desktop_auto_execute: bool,
    desktop_auto_approve_raw: str,
    desktop_max_steps: int,
) -> None:
    allowed_paths = [
        line.strip() for line in allowed_paths_raw.splitlines() if line.strip()
    ]
    effective_run_mode = (
        "desktop" if _should_use_desktop_flow(description, run_mode) else run_mode
    )

    if effective_run_mode == "desktop":
        _, graph = build_desktop_demo_graph(
            env_file=env_file or ".env",
            workspace_root=".",
            include_starter_agents=True,
            include_critic=include_critic,
            include_planner=use_planner,
            include_validator=validation_mode == FactLogicValidationMode.SPECIALIST,
        )
        task = build_desktop_task(
            query=description,
            allowed_paths=allowed_paths,
            hitl_points=_selected_hitl_points(confirm_before_action),
            locale="ru",
        )
        desktop_context = {
            "current_path": allowed_paths[0] if allowed_paths else ".",
            "installed_packages": [],
            "trust_score": 0.5,
            "last_actions": [],
        }
        if (
            desktop_auto_execute
            and task.assignee == "planner"
            and isinstance(task.metadata.get("subtasks"), list)
        ):
            auto_approve_tools = [
                item.strip()
                for item in desktop_auto_approve_raw.splitlines()
                if item.strip()
            ]
            plan_execution = run_desktop_plan(
                graph,
                task=task,
                stream=use_stream,
                desktop_context=desktop_context,
                auto_approve_tools=auto_approve_tools,
                max_steps=desktop_max_steps,
            )
            state = plan_execution.step_states[-1]
            events = [event for batch in plan_execution.step_events for event in batch]
            st.session_state.plan_execution = {
                "root_thread_id": plan_execution.root_thread_id,
                "steps_completed": len(plan_execution.step_states),
                "stopped_reason": plan_execution.stopped_reason,
                "step_statuses": [
                    step.status.value for step in plan_execution.step_states
                ],
                "execution": plan_execution,
                "desktop_context": desktop_context,
                "auto_approve_tools": auto_approve_tools,
                "task": task,
            }
        else:
            state, events = run_desktop_task(
                graph,
                task=task,
                stream=use_stream,
                desktop_context=desktop_context,
            )
            st.session_state.plan_execution = None
    else:
        _, graph = build_demo_graph(
            env_file=env_file or ".env",
            workspace_root=".",
            include_critic=include_critic,
        )
        task = build_demo_task(
            query=description,
            enable_planner=use_planner,
            fact_logic_validation=validation_mode,
            hitl_points=_selected_hitl_points(confirm_before_action),
            locale="ru",
        )
        state, events = run_task(graph, task=task, stream=use_stream)
        st.session_state.plan_execution = None

    st.session_state.graph = graph
    st.session_state.last_task = task
    st.session_state.run_mode = effective_run_mode
    st.session_state.run_result = {
        "thread_id": state.thread_id,
        "state": state,
        "events": events,
    }
    st.session_state.events = events
    st.session_state.last_error = None
    st.session_state.export_result = None


def _resume_run(decision: str, note: str) -> None:
    graph = st.session_state.graph
    run_result = st.session_state.run_result
    if graph is None or run_result is None:
        st.warning("Сначала запустите задачу.")
        return

    human_feedback: dict[str, Any] = {"decision": decision}
    if note.strip():
        human_feedback["note"] = note.strip()

    if st.session_state.run_mode == "desktop":
        plan_execution = st.session_state.plan_execution
        if (
            isinstance(plan_execution, dict)
            and plan_execution.get("execution") is not None
            and plan_execution.get("task") is not None
        ):
            resumed_plan = resume_desktop_plan(
                graph,
                task=plan_execution["task"],
                paused_state=run_result["state"],
                human_feedback=human_feedback,
                desktop_context=plan_execution.get("desktop_context"),
                root_thread_id=plan_execution.get("root_thread_id"),
                auto_approve_tools=plan_execution.get("auto_approve_tools"),
                prior_step_states=plan_execution["execution"].step_states,
                prior_step_events=plan_execution["execution"].step_events,
            )
            state = resumed_plan.step_states[-1]
            events = [event for batch in resumed_plan.step_events for event in batch]
            st.session_state.plan_execution = {
                "root_thread_id": resumed_plan.root_thread_id,
                "steps_completed": len(resumed_plan.step_states),
                "stopped_reason": resumed_plan.stopped_reason,
                "step_statuses": [
                    step.status.value for step in resumed_plan.step_states
                ],
                "execution": resumed_plan,
                "desktop_context": plan_execution.get("desktop_context"),
                "auto_approve_tools": plan_execution.get("auto_approve_tools"),
                "task": plan_execution.get("task"),
            }
        else:
            state, events = resume_desktop_task(
                graph,
                thread_id=run_result["thread_id"],
                human_feedback=human_feedback,
            )
    else:
        state, events = resume_task(
            graph,
            thread_id=run_result["thread_id"],
            human_feedback=human_feedback,
        )

    st.session_state.run_result = {
        "thread_id": state.thread_id,
        "state": state,
        "events": events,
    }
    st.session_state.events = events
    st.session_state.last_error = None


def _export_last_desktop_artifact() -> None:
    run_result = st.session_state.run_result
    graph = st.session_state.graph
    if graph is None or run_result is None:
        st.session_state.last_error = "Нет активного desktop-результата для экспорта."
        return

    state = run_result["state"]
    result = export_desktop_artifact(graph, state=state)
    st.session_state.export_result = result
    if result.success:
        st.session_state.last_error = None
    else:
        st.session_state.last_error = result.error or "Не удалось выполнить экспорт."


def _render_sidebar() -> dict[str, Any]:
    st.sidebar.title("Параметры")
    st.sidebar.caption("Настройте запуск задачи без технических деталей.")

    run_mode_label = st.sidebar.radio(
        "Режим",
        options=list(RUN_MODE_OPTIONS.keys()),
        index=0,
        help=(
            "Обычный режим подходит для исследования и ответов. "
            "Desktop Assistant нужен для задач с файлами, загрузками "
            "и приложениями."
        ),
    )
    run_mode = RUN_MODE_OPTIONS[run_mode_label]

    st.sidebar.markdown("### Как работать с задачей")
    use_planner = st.sidebar.checkbox(
        "Разбивать сложную задачу на шаги",
        value=False,
        help="Полезно для больших задач с несколькими этапами.",
    )
    use_stream = st.sidebar.checkbox(
        "Показывать ход выполнения по шагам",
        value=True,
        help="Если включено, интерфейс покажет историю выполнения.",
    )

    st.sidebar.markdown("### Когда спрашивать подтверждение")
    confirm_before_action = st.sidebar.checkbox(
        "Запрашивать подтверждение перед действиями с файлами и внешними системами",
        value=True,
        help="Рекомендуется оставлять включённым.",
    )
    validation_mode_label = st.sidebar.selectbox(
        "Насколько строго проверять результат",
        options=list(VALIDATION_MODE_OPTIONS.keys()),
        index=1,
        help="Можно оставить стандартный режим с правилами проверки.",
    )

    with st.sidebar.expander("Продвинутые настройки"):
        env_file = st.text_input(
            "Путь к .env",
            value=str(Path.cwd().parents[1] / ".env"),
            help="Укажите путь, если переменные окружения лежат не в корневом `.env`.",
        )
        include_critic = st.checkbox(
            "Подключать критика для поиска слабых мест",
            value=False,
        )
        allowed_paths_raw = st.text_area(
            "Разрешённые пути для Desktop Assistant",
            value="~/Desktop\n~/Documents\n~/Downloads",
            height=100,
            help="Используется только в desktop-режиме.",
        )

    with st.sidebar.expander("Какие агенты участвуют"):
        st.json(_build_agent_overview(include_critic, run_mode))

    return {
        "run_mode": run_mode,
        "env_file": env_file,
        "use_planner": use_planner,
        "validation_mode": VALIDATION_MODE_OPTIONS[validation_mode_label],
        "confirm_before_action": confirm_before_action,
        "include_critic": include_critic,
        "use_stream": use_stream,
        "allowed_paths_raw": allowed_paths_raw,
        "desktop_auto_execute": True,
        "desktop_auto_approve_raw": "fs.create_dir\nfs.move\nfs.delete\nweb.search\nweb.download\napp.launch",
        "desktop_max_steps": 5,
    }


def _render_human_checkpoint(state: Any) -> None:
    checkpoint = state.human_checkpoint
    if checkpoint is None:
        return

    checkpoint_data = checkpoint.model_dump(mode="json")
    reason = checkpoint_data.get("reason", "")
    reason_label = CHECKPOINT_REASON_LABELS.get(reason, "нужно ваше решение")
    st.info(f"Система ждёт ваше решение: {reason_label}.")

    preview = checkpoint_data.get("request", {}).get(
        "dry_run_preview"
    ) or state.task.metadata.get("dry_run_preview")
    if preview:
        with st.expander("Что именно планируется сделать"):
            st.code(str(preview))

    risk_level = state.task.metadata.get("risk_level")
    quarantine_status = state.task.metadata.get("quarantine_status")
    allowed_paths = state.task.metadata.get("allowed_paths") or []

    if risk_level:
        st.caption(f"Риск: {risk_level}")
    if quarantine_status:
        st.caption(f"Карантин: {quarantine_status}")
    if allowed_paths:
        st.caption(
            "Разрешённые пути: " + ", ".join(str(path) for path in allowed_paths)
        )
    if reason == "tool_approval":
        st.warning("После подтверждения действие будет выполнено реально.")


def _render_result_summary(state: Any) -> None:
    output = state.output_candidate or {}
    summary = output.get("summary") if isinstance(output, dict) else None
    confidence = output.get("confidence") if isinstance(output, dict) else None

    if summary:
        st.markdown("### Результат")
        st.write(summary)
    elif state.status == ThreadStatus.COMPLETED:
        st.markdown("### Результат")
        st.write(
            "Задача завершена, но краткое текстовое описание не было сформировано."
        )

    if confidence is not None:
        try:
            st.caption(f"Уверенность системы: {float(confidence):.0%}")
        except (TypeError, ValueError):
            pass


def _render_current_state() -> None:
    run_result = st.session_state.run_result
    if run_result is None:
        st.info("Введите задачу и нажмите «Запустить», чтобы начать проверку системы.")
        return

    state = run_result["state"]
    status_label = STATUS_LABELS.get(state.status, str(state.status))

    st.subheader("Краткая сводка")
    if state.status == ThreadStatus.COMPLETED:
        st.success(status_label)
    elif state.status == ThreadStatus.HITL_WAIT:
        st.warning(status_label)
    elif state.status == ThreadStatus.FAILED:
        st.error(status_label)
    else:
        st.info(status_label)

    if state.current_agent:
        agent_label = AGENT_LABELS.get(state.current_agent, state.current_agent)
        st.caption(f"Сейчас активен: {agent_label}")

    retries = sum(int(value) for value in state.retry_counters.values())
    if retries > 0:
        st.caption(f"Повторных попыток было: {retries}")
    else:
        st.caption("Повторных попыток пока не было.")

    desktop_context = state.shared_context.get("desktop_context")
    if isinstance(desktop_context, dict) and st.session_state.run_mode == "desktop":
        current_path = desktop_context.get("current_path")
        trust_score = desktop_context.get("trust_score")
        if current_path:
            st.caption(f"Текущий рабочий путь: {current_path}")
        if trust_score is not None:
            try:
                st.caption(f"Уровень доверия: {float(trust_score):.0%}")
            except (TypeError, ValueError):
                pass

    can_export = (
        st.session_state.run_mode == "desktop"
        and state.status == ThreadStatus.COMPLETED
        and state.task.metadata.get("tool_ref") in {"fs.create_dir", "web.download"}
        and bool(state.task.metadata.get("requested_path"))
    )
    if can_export:
        st.info(
            "\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0441\u0435\u0439\u0447\u0430\u0441 "
            "\u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u0435\u0442 "
            "\u0442\u043e\u043b\u044c\u043a\u043e \u0432 \u043f\u0435\u0441\u043e\u0447\u043d\u0438\u0446\u0435. "
            "\u0427\u0442\u043e\u0431\u044b \u043f\u0435\u0440\u0435\u043d\u0435\u0441\u0442\u0438 \u0435\u0433\u043e "
            "\u0432 \u0440\u0435\u0430\u043b\u044c\u043d\u043e\u043c "
            "\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c\u0441\u043a\u043e\u043c "
            "\u043f\u0443\u0442\u0438, \u0432\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u0435 "
            "\u044d\u043a\u0441\u043f\u043e\u0440\u0442."
        )
        if st.button(
            "\U0001f4e5 \u042d\u043a\u0441\u043f\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c "
            "\u0432 \u0438\u0441\u0445\u043e\u0434\u043d\u044b\u0439 \u043f\u0443\u0442\u044c",
            use_container_width=True,
        ):
            _export_last_desktop_artifact()
            st.rerun()

    export_result = st.session_state.export_result
    if export_result is not None:
        if export_result.success:
            exported_path = export_result.data.get("exported_path")
            st.success(
                f"\u042d\u043a\u0441\u043f\u043e\u0440\u0442 "
                f"\u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d: {exported_path}"
            )
        else:
            st.error(
                export_result.error
                or "\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c "
                "\u0432\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c \u044d\u043a\u0441\u043f\u043e\u0440\u0442."
            )

    _render_human_checkpoint(state)
    _render_result_summary(state)

    if state.output_candidate:
        with st.expander("Показать структурированный результат"):
            st.json(state.output_candidate)

    if state.memory_refs:
        with st.expander("Показать сохранённые ссылки и память"):
            st.json(state.memory_refs)

    if st.session_state.events:
        with st.expander("Показать ход выполнения"):
            st.json(st.session_state.events)

    if state.errors:
        with st.expander("Показать ошибки и замечания"):
            st.json(state.errors)


def _render_resume_form() -> None:
    run_result = st.session_state.run_result
    if run_result is None or run_result["state"].status != ThreadStatus.HITL_WAIT:
        return

    st.markdown("### Ваше решение")
    with st.form("resume_form"):
        decision = st.radio(
            "Что сделать дальше",
            options=["approve", "amend", "reject"],
            format_func=lambda value: {
                "approve": "Подтвердить",
                "amend": "Подтвердить с комментарием",
                "reject": "Отклонить",
            }[value],
        )
        note = st.text_area(
            "Комментарий",
            value="",
            help="Можно оставить пояснение для системы.",
        )
        submitted = st.form_submit_button("Продолжить")

    if submitted:
        _resume_run(decision, note)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="AgentGraph Demo", layout="wide")
    st.title("Локальный тест AgentGraph")
    st.caption(
        "Интерфейс для проверки агентов, маршрутизации, подтверждений и desktop-режима."
    )

    _init_session_state()
    settings = _render_sidebar()
    description = st.text_area(
        "Что нужно сделать",
        value="Собери краткое описание того, для чего используется LangGraph.",
        height=140,
        help="Опишите задачу простыми словами.",
    )

    if st.button("Запустить", type="primary", use_container_width=True):
        try:
            _start_run(description=description, **settings)
        except Exception as exc:  # pragma: no cover - UI surface
            st.session_state.last_error = str(exc)
            st.error(f"Не удалось запустить задачу: {exc}")

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    _render_current_state()
    _render_resume_form()

    with st.expander("Сырой снимок session state"):
        raw_snapshot = {
            "run_mode": st.session_state.run_mode,
            "thread_id": (
                st.session_state.run_result["thread_id"]
                if st.session_state.run_result is not None
                else None
            ),
            "run_result": (
                st.session_state.run_result["state"].model_dump(mode="json")
                if st.session_state.run_result is not None
                else None
            ),
            "events": st.session_state.events,
            "last_error": st.session_state.last_error,
        }
        st.code(
            json.dumps(raw_snapshot, ensure_ascii=False, indent=2, default=str),
            language="json",
        )


if __name__ == "__main__":
    main()
