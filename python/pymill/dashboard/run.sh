rm -r resultsviewer/migrations
python manage.py makemigrations resultsviewer
python manage.py migrate --database=results
python manage.py migrate
python manage.py runserver

