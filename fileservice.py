from fastapi import FastAPI, Request, Response

filenames = ["js/interactive_grid.js"]
contents = "\n".join(
    [f"<script type='text/javascript' src='{x}'></script>" for x in filenames]
)

ga_script = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-11ZHMNWP9Y"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-11ZHMNWP9Y');
</script>
"""

app = FastAPI()


@app.middleware("http")
async def insert_js(request: Request, call_next):
    path = request.scope["path"]  # get the request route
    response = await call_next(request)

    if path == "/":
        response_body = ""
        async for chunk in response.body_iterator:
            response_body += chunk.decode()

        charset_tag = '<meta charset="utf-8" />'
        if charset_tag in response_body:
            response_body = response_body.replace(charset_tag, charset_tag + ga_script)

        response_body = response_body.replace("</body>", contents + "</body>")

        del response.headers["content-length"]

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    return response
