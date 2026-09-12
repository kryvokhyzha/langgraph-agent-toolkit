@ECHO OFF
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1

REM Use the installed locked environment. Keep caller overrides.
if not defined SPHINXBUILD set "SPHINXBUILD=uv run --no-sync sphinx-build"
if not defined SPHINXAPIDOC set "SPHINXAPIDOC=uv run --no-sync sphinx-apidoc"
if not defined SPHINXOPTS set "SPHINXOPTS=-W --keep-going"
set "SOURCEDIR=."
set "BUILDDIR=_build"
set "DOC_TARGET=%~1"
if not defined DOC_TARGET set "DOC_TARGET=help"

if /I "%DOC_TARGET%" == "api" goto api
if /I "%DOC_TARGET%" == "html" (
    set "DOC_TARGET=html"
    call :generate_api
    if errorlevel 1 goto end
)

call %SPHINXBUILD% -M "%DOC_TARGET%" "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
goto end

:api
call :generate_api
goto end

:generate_api
REM Generate the API root and remove pages for deleted modules.
call %SPHINXAPIDOC% -f -e -M --remove-old -o generated ../langgraph_agent_toolkit --doc-project="API Reference"
exit /b %ERRORLEVEL%

:end
set "DOC_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %DOC_EXIT_CODE%
