"""Tests for class-level dependency declarations on Dependency subclasses."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import pytest

from uncalled_for.classy import (  # pyright: ignore[reportPrivateUsage]
    _ContextSensitiveAttr,
    _unwrap,
)

from uncalled_for import (
    Dependency,
    Depends,
    FailedDependency,
    Shared,
    SharedContext,
    get_class_dependencies,
    resolved_dependencies,
    without_dependencies,
)


class Greeter(Dependency[str]):
    async def __aenter__(self) -> str: ...


def test_class_without_class_deps_unchanged() -> None:
    assert not hasattr(Greeter, "__class_dependencies__")
    assert get_class_dependencies(Greeter) == {}


async def test_basic_depends_class_attr() -> None:
    def get_value() -> str:
        return "resolved-value"

    class MyDep(Dependency[str]):
        value: str = Depends(get_value)

        async def __aenter__(self) -> str:
            return f"got {self.value}"

    async with MyDep() as result:
        assert result == "got resolved-value"


async def test_bare_dependency_subclass_as_class_attr() -> None:
    class Inner(Dependency[int]):
        async def __aenter__(self) -> int:
            return 42

    class Outer(Dependency[str]):
        inner: int = Inner()  # type: ignore[assignment]

        async def __aenter__(self) -> str:
            return f"inner={self.inner}"

    async with Outer() as result:
        assert result == "inner=42"


async def test_self_typed_dependency_as_class_attr() -> None:
    class Config(Dependency["Config"]):
        def __init__(self, url: str = "http://default") -> None:
            self.url = url

        async def __aenter__(self) -> Config:
            return self

    class Service(Dependency[str]):
        config: Config = Config()

        async def __aenter__(self) -> str:
            return f"service({self.config.url})"

    async with Service() as result:
        assert result == "service(http://default)"


async def test_multiple_class_level_deps() -> None:
    def get_a() -> str:
        return "a"

    def get_b() -> int:
        return 2

    class Multi(Dependency[str]):
        a: str = Depends(get_a)
        b: int = Depends(get_b)

        async def __aenter__(self) -> str:
            return f"{self.a}-{self.b}"

    async with Multi() as result:
        assert result == "a-2"


async def test_lifecycle_ordering() -> None:
    order: list[str] = []

    @asynccontextmanager
    async def resource_a() -> AsyncGenerator[str]:
        order.append("a-enter")
        yield "a"
        order.append("a-exit")

    @asynccontextmanager
    async def resource_b() -> AsyncGenerator[str]:
        order.append("b-enter")
        yield "b"
        order.append("b-exit")

    class LifecycleDep(Dependency[str]):
        a: str = Depends(resource_a)
        b: str = Depends(resource_b)

        async def __aenter__(self) -> str:
            order.append("enter")
            return f"{self.a},{self.b}"

        async def __aexit__(self, *args: object) -> None:
            order.append("exit")

    async with LifecycleDep() as result:
        assert result == "a,b"

    assert order == ["a-enter", "b-enter", "enter", "exit", "b-exit", "a-exit"]


async def test_integration_with_resolved_dependencies() -> None:
    def get_connection() -> str:
        return "db-conn"

    class DbDep(Dependency[str]):
        conn: str = Depends(get_connection)

        async def __aenter__(self) -> str:
            return f"using {self.conn}"

    def make_db_dep() -> str:
        return cast(str, DbDep())

    async def my_func(db: str = make_db_dep()) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps["db"] == "using db-conn"


async def test_integration_with_without_dependencies() -> None:
    def get_connection() -> str:
        return "db-conn"

    class DbDep(Dependency[str]):
        conn: str = Depends(get_connection)

        async def __aenter__(self) -> str:
            return f"using {self.conn}"

    def make_db_dep() -> str:
        return cast(str, DbDep())

    async def my_func(db: str = make_db_dep()) -> str:
        return db

    wrapped = without_dependencies(my_func)
    result = await wrapped()
    assert result == "using db-conn"


async def test_shared_class_attr() -> None:
    call_count = 0

    def make_pool() -> str:
        nonlocal call_count
        call_count += 1
        return "shared-pool"

    class PoolDep(Dependency[str]):
        pool: str = Shared(make_pool)

        async def __aenter__(self) -> str:
            return self.pool

    def make_pool_dep() -> str:
        return cast(str, PoolDep())

    async def func_a(v: str = make_pool_dep()) -> None: ...
    async def func_b(v: str = make_pool_dep()) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(func_a) as deps_a:
            assert deps_a["v"] == "shared-pool"

        async with resolved_dependencies(func_b) as deps_b:
            assert deps_b["v"] == "shared-pool"

        assert call_count == 1


async def test_error_in_class_dep_propagates() -> None:
    class Broken(Dependency[str]):
        async def __aenter__(self) -> str:
            raise RuntimeError("broken dep")

    class Consumer(Dependency[str]):
        broken: str = Depends(Broken)

        async def __aenter__(self) -> str: ...

    with pytest.raises(RuntimeError, match="broken dep"):
        async with Consumer():
            ...


async def test_error_in_class_dep_cleans_up_others() -> None:
    cleaned_up = False

    @asynccontextmanager
    async def good_resource() -> AsyncGenerator[str]:
        yield "good"
        nonlocal cleaned_up
        cleaned_up = True

    class Broken(Dependency[str]):
        async def __aenter__(self) -> str:
            raise RuntimeError("broken")

    class Consumer(Dependency[str]):
        good: str = Depends(good_resource)
        broken: str = Depends(Broken)

        async def __aenter__(self) -> str: ...

    with pytest.raises(RuntimeError, match="broken"):
        async with Consumer():
            ...

    assert cleaned_up


async def test_error_in_aenter_cleans_up_deps() -> None:
    cleaned_up = False

    @asynccontextmanager
    async def managed_resource() -> AsyncGenerator[str]:
        yield "resource"
        nonlocal cleaned_up
        cleaned_up = True

    class FailingDep(Dependency[str]):
        resource: str = Depends(managed_resource)

        async def __aenter__(self) -> str:
            raise RuntimeError("aenter failed")

    with pytest.raises(RuntimeError, match="aenter failed"):
        async with FailingDep():
            ...

    assert cleaned_up


async def test_error_inside_resolved_dependencies_context() -> None:
    class Broken(Dependency[str]):
        async def __aenter__(self) -> str:
            raise RuntimeError("boom")

    class Consumer(Dependency[str]):
        broken: str = Depends(Broken)

        async def __aenter__(self) -> str: ...

    def make_consumer() -> str:
        return cast(str, Consumer())

    async def my_func(c: str = make_consumer()) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert isinstance(deps["c"], FailedDependency)
        assert isinstance(deps["c"].error, RuntimeError)


def test_accessing_dep_before_aenter_raises() -> None:
    def get_value() -> str: ...

    class MyDep(Dependency[str]):
        value: str = Depends(get_value)

        async def __aenter__(self) -> str: ...

    instance = MyDep()
    with pytest.raises(AttributeError):
        instance.value  # noqa: B018


def test_get_class_dependencies_returns_deps() -> None:
    class Inner(Dependency[int]):
        async def __aenter__(self) -> int: ...

    class MyDep(Dependency[str]):
        x: int = Depends(Inner)

        async def __aenter__(self) -> str: ...

    deps = get_class_dependencies(MyDep)
    assert "x" in deps
    assert isinstance(deps["x"], Dependency)

    assert isinstance(MyDep.x, _ContextSensitiveAttr)


async def test_standalone_usage_without_resolved_dependencies() -> None:
    def get_value() -> str:
        return "standalone"

    class StandaloneDep(Dependency[str]):
        value: str = Depends(get_value)

        async def __aenter__(self) -> str:
            return self.value

    async with StandaloneDep() as result:
        assert result == "standalone"


def test_unwrap_raises_for_missing_method() -> None:
    class Empty:
        pass

    with pytest.raises(TypeError, match="Empty has no __aenter__"):
        _unwrap(Empty, "__aenter__", "__original__")  # type: ignore[arg-type]


async def test_class_dep_values_isolated_across_concurrent_entry() -> None:
    call_count = 0

    def make_value() -> str:
        nonlocal call_count
        call_count += 1
        return f"value-{call_count}"

    class SelfReturning(Dependency["SelfReturning"]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> SelfReturning:
            return self

    shared_instance = SelfReturning()
    entered = asyncio.Event()
    proceed = asyncio.Event()
    values: dict[str, str] = {}

    async def task_a() -> None:
        async with shared_instance as dependency:
            values["a_initial"] = dependency.value
            entered.set()
            await proceed.wait()
            values["a_after_b"] = dependency.value

    async def task_b() -> None:
        await entered.wait()
        async with shared_instance as dependency:
            values["b"] = dependency.value
            proceed.set()

    await asyncio.gather(task_a(), task_b())

    assert values["a_initial"] != values["b"]
    assert values["a_initial"] == values["a_after_b"]


async def test_concurrent_entry_cleanup_is_isolated() -> None:
    cleanups: list[str] = []

    @asynccontextmanager
    async def tracked_resource() -> AsyncGenerator[str]:
        task = asyncio.current_task()
        name = task.get_name() if task else "unknown"
        yield name
        cleanups.append(name)

    class TrackedDependency(Dependency[str]):
        resource: str = Depends(tracked_resource)

        async def __aenter__(self) -> str:
            return self.resource

    shared_instance = TrackedDependency()
    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def task_a() -> None:
        async with shared_instance:
            entered.set()
            await proceed.wait()

    async def task_b() -> None:
        await entered.wait()
        async with shared_instance:
            proceed.set()

    a_task = asyncio.ensure_future(task_a())
    a_task.set_name("a")
    b_task = asyncio.ensure_future(task_b())
    b_task.set_name("b")

    await asyncio.gather(a_task, b_task)

    assert set(cleanups) == {"a", "b"}


async def test_inherited_class_deps_isolated_across_concurrent_entry() -> None:
    call_count = 0

    def make_value() -> str:
        nonlocal call_count
        call_count += 1
        return f"value-{call_count}"

    class Parent(Dependency["Parent"]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> Parent:
            return self

    class Child(Parent):
        pass

    shared_instance = Child()
    entered = asyncio.Event()
    proceed = asyncio.Event()
    values: dict[str, str] = {}

    async def task_a() -> None:
        async with shared_instance as dependency:
            values["a_initial"] = dependency.value
            entered.set()
            await proceed.wait()
            values["a_after_b"] = dependency.value

    async def task_b() -> None:
        await entered.wait()
        async with shared_instance as dependency:
            values["b"] = dependency.value
            proceed.set()

    await asyncio.gather(task_a(), task_b())

    assert values["a_initial"] != values["b"]
    assert values["a_initial"] == values["a_after_b"]


async def test_nested_class_deps_isolated_across_concurrent_entry() -> None:
    call_count = 0

    def make_value() -> str:
        nonlocal call_count
        call_count += 1
        return f"value-{call_count}"

    class Inner(Dependency["Inner"]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> Inner:
            return self

    class Outer(Dependency["Outer"]):
        inner: Inner = Depends(Inner)

        async def __aenter__(self) -> Outer:
            return self

    shared_instance = Outer()
    entered = asyncio.Event()
    proceed = asyncio.Event()
    values: dict[str, str] = {}

    async def task_a() -> None:
        async with shared_instance as dependency:
            values["a_initial"] = dependency.inner.value
            entered.set()
            await proceed.wait()
            values["a_after_b"] = dependency.inner.value

    async def task_b() -> None:
        await entered.wait()
        async with shared_instance as dependency:
            values["b"] = dependency.inner.value
            proceed.set()

    await asyncio.gather(task_a(), task_b())

    assert values["a_initial"] != values["b"]
    assert values["a_initial"] == values["a_after_b"]


async def test_concurrent_error_does_not_corrupt_healthy_task() -> None:
    entered = asyncio.Event()
    may_fail = asyncio.Event()

    def make_value() -> str:
        return "healthy"

    class ErrorDependency(Dependency[str]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> str:
            return self.value

    shared_instance = ErrorDependency()
    healthy_result: str | None = None

    async def healthy_task() -> None:
        nonlocal healthy_result
        async with shared_instance as result:
            healthy_result = result
            entered.set()
            await may_fail.wait()
            healthy_result = result

    async def failing_task() -> None:
        await entered.wait()
        async with shared_instance:
            may_fail.set()
            raise RuntimeError("boom")

    results = await asyncio.gather(
        healthy_task(), failing_task(), return_exceptions=True
    )

    assert healthy_result == "healthy"
    assert isinstance(results[1], RuntimeError)


async def test_many_concurrent_tasks_isolated() -> None:
    call_count = 0

    def make_value() -> str:
        nonlocal call_count
        call_count += 1
        return f"value-{call_count}"

    class SelfReturning(Dependency["SelfReturning"]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> SelfReturning:
            return self

    shared_instance = SelfReturning()
    task_count = 20
    barrier = asyncio.Barrier(task_count)
    values: dict[int, str] = {}

    async def worker(index: int) -> None:
        async with shared_instance as dependency:
            values[index] = dependency.value
            await barrier.wait()
            assert dependency.value == values[index]

    await asyncio.gather(*(worker(i) for i in range(task_count)))
