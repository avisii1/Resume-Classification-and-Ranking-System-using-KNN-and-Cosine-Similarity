from flask import Flask, render_template, request, redirect, url_for, session, flash,send_from_directory,make_response
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import os
from flask_bcrypt import Bcrypt
from pdfextractor import extract_text_from_pdf
from preprocessor import preprocess
from knn_algo import KNN
from tfidf import TFIDF
from labelencoder import CustomLabelEncoder
from cosineSimilarity import cosine_similarity
import numpy as np
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'XYZ123'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3307
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'rcrs'

mysql = MySQL(app)
bcrypt = Bcrypt(app)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load your trained model
with open('knn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizerTFIDF.pkl', 'rb') as model_file:
    vectorizer = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as model_file:
    encoder = pickle.load(model_file)


def extract_text_and_classify(file_path, job_id):
    # Extract text from the PDF
    text = extract_text_from_pdf(file_path)  

    # Preprocess the text
    preprocessed_text = preprocess(text)  

    # TF-IDF transformation
    tfidf_matrix = vectorizer.transform([preprocessed_text])

    # Predict using the trained KNN model
    predicted_category = model.predict(tfidf_matrix)

    cur = mysql.connection.cursor()
    cur.execute("SELECT title, description, requirements from job_postings where id = %s",
    (job_id,))
    result = cur.fetchone()
    mysql.connection.commit()
    cur.close()

    if result:
        title, description, requirements = result
        
        # Concatenate the title, description, and requirements
        job_text = f"{title}\n\n{description}\n\n{requirements}"

    preprocessed_job_text = preprocess(job_text)

    job_tfidf = vectorizer.transform([preprocessed_job_text])

    tfidf_matrix = np.array(tfidf_matrix)
    job_tfidf = np.array(job_tfidf)

    # Simulated cosine similarity score
    similarity_score = cosine_similarity(tfidf_matrix, job_tfidf)

    return predicted_category[0], similarity_score[0][0]


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# ...
############################################
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def home():
    return render_template('base.html')
# @app.route('/')
# def home():
#     if 'user_id' in session:
#         user_email = session['user_id']
#         cur = mysql.connection.cursor()

#         # Check if the user is an admin
#         result_admin = cur.execute("SELECT * FROM admins WHERE email = %s", [user_email])

#         if result_admin > 0:
#             return render_template('admin_panel.html')

#         # If not an admin, assume it's a candidate and redirect to home
#         return render_template('base.html')
#     else:
#     # If user_id is not in session, redirect to login
#         return redirect(url_for('login'))
#############################################

@app.route('/admin_panel')
def admin_panel():
    if 'user_id' in session:    
        if 'user_id' in session:
            user_email = session['user_id']
            cur = mysql.connection.cursor()

            # Check if the user is an admin
            result_admin = cur.execute("SELECT * FROM admins WHERE email = %s", [user_email])

            if result_admin > 0:
                return render_template('admin_panel.html')
            
        return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))
    

# Add these new routes

@app.route('/create_job_posting', methods=['GET', 'POST'])
def create_job_posting():
    if 'user_id' in session:
        if request.method == 'POST':
            title = request.form['title']
            description = request.form['description']
            requirements = request.form['requirements']
            salary = request.form['salary']

            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO job_postings (title, description, requirements, salary) VALUES (%s, %s, %s, %s)",
                (title, description, requirements, salary))
            mysql.connection.commit()
            cur.close()

            flash('Job posting created successfully', 'success')

        return render_template('create_job_posting.html')
    else :
        return redirect(url_for('login'))

@app.route('/view_jobs')
def view_jobs():
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM job_postings")
        job_postings = cur.fetchall()
        cur.close()

        return render_template('view_jobs.html', job_postings=job_postings)
    
#for admin view job
@app.route('/view_job_admin')
def view_job_admin():
    if 'user_id' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM job_postings")
        job_postings = cur.fetchall()
        cur.close()

        return render_template('view_job_admin.html', job_postings=job_postings)
    else:
        return redirect(url_for('login'))
#admin job details
# ...

@app.route('/job_detail/<int:job_id>')
def job_detail(job_id):
    if 'user_id' in session:
        user_email = session['user_id']
        cur = mysql.connection.cursor()

        # Check if the user is an admin
        result_admin = cur.execute("SELECT * FROM admins WHERE email = %s", [user_email])

        if result_admin > 0:
            query = """
                SELECT job_applications.id AS application_id, 
                    job_postings.id AS job_id, 
                    job_postings.title AS job_title, 
                    candidates.full_name, 
                    candidates.email, 
                    resumes.file_path
                FROM job_applications
                JOIN job_postings ON job_applications.job_posting_id = job_postings.id
                JOIN candidates ON job_applications.candidate_id = candidates.id
                JOIN resumes ON job_applications.candidate_id = resumes.candidate_id
                WHERE job_postings.id = %s
                """
            cur.execute(query, (job_id,))
            job_applications = cur.fetchone()
            cur.close()

            return render_template('job_applications.html', job_applications=job_applications)

        return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))

# ...


@app.route('/view_job/<int:job_id>')
def view_job(job_id):
    # if 'user_id' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM job_postings WHERE id = %s", [job_id])
        job_posting = cur.fetchone()
        cur.close()

        return render_template('view_job.html', job_posting=job_posting)
    # else :
    #     return redirect(url_for('login'))

@app.route('/apply_job/<int:job_id>', methods=['GET', 'POST'])
def apply_job(job_id):
    if 'user_id' in session:
        user_email = session['user_id']
        #####################################################
                # Check if the candidate has already applied for this job posting
        cur = mysql.connection.cursor()
        result = cur.execute(
            "SELECT * FROM job_applications WHERE candidate_id = (SELECT id FROM candidates WHERE email = %s) AND job_posting_id = %s",
            (user_email, job_id)
        )
        
        if result > 0:
            # Candidate has already applied, show a message or redirect as needed
            flash('You have already applied for this job.', 'error')
            return redirect(url_for('view_job', job_id=job_id))

        #################################################

        if request.method == 'POST':
            # Check if the post request has the file part
            if 'resume' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)

            resume_file = request.files['resume']

            # If the user does not select a file, the browser will submit an empty part without a filename
            if resume_file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)

            # Check if the file extension is allowed
            if resume_file and allowed_file(resume_file.filename):
                # Save the resume to the uploads folder
                filename = secure_filename(resume_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume_file.save(file_path)

                # Store the resume information in the database
                cur = mysql.connection.cursor()
                cur.execute(
                    "INSERT INTO resumes (candidate_id, file_path) VALUES ((SELECT id FROM candidates WHERE email = %s), %s)",
                    (user_email, filename))
                mysql.connection.commit()
                cur.close()

                # Fetching the newly inserted resume_id
                cur = mysql.connection.cursor()
                cur.execute("SELECT LAST_INSERT_ID()")
                resume_id = cur.fetchone()[0]
                cur.close()

                #store the information in application table
                cur = mysql.connection.cursor()
                cur.execute(
                    "INSERT INTO job_applications (job_posting_id, candidate_id, resume_id) VALUES (%s, (SELECT id FROM candidates WHERE email = %s), %s)",
                    (job_id, user_email, resume_id))
                mysql.connection.commit()
                cur.close()

                flash('Resume submitted successfully', 'success')
                return redirect(url_for('view_job', job_id=job_id))

        return render_template('apply_job.html', job_id=job_id)

    else:
        flash('Please login to apply for the job', 'error')
         
        return redirect(url_for('view_job', job_id=job_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ...

#fetching data for displaying in job_application page
def get_job_applications_data(job_id):
    connection = mysql.connection
    cursor = connection.cursor()

    query = """
            SELECT
            job_applications.id AS application_id,
            job_applications.resume_id as r_id,
            job_postings.id AS job_id,
            job_postings.title AS job_title,
            candidates.full_name,
            candidates.email,
            resumes.file_path
        FROM
            job_applications
        JOIN job_postings ON job_applications.job_posting_id = job_postings.id
        JOIN candidates ON job_applications.candidate_id = candidates.id
        JOIN resumes ON job_applications.resume_id = resumes.id
        WHERE
            job_applications.job_posting_id = %s;
    """
    
    cursor.execute(query,(job_id,))

    columns = [column[0] for column in cursor.description]
    job_applications_data = [dict(zip(columns, row)) for row in cursor.fetchall()]

    # Classify and store results in the database
    for application in job_applications_data:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], application['file_path'])
        predicted_category, similarity_score = extract_text_and_classify(file_path, application['job_id'])
        predicted_category = encoder.inverse_transform(predicted_category)

        cursor.execute("""
            SELECT COUNT(*) FROM classified_resumes WHERE file_path = %s
        """, (application['file_path'],))
        existing_record = cursor.fetchone()[0]

        if existing_record:
            # Update existing record
            cursor.execute("""
                UPDATE classified_resumes 
                SET classified_category = %s, similarity_score = %s 
                WHERE file_path = %s
            """, (predicted_category, similarity_score, application['file_path']))
        else:
            # Insert new record
            cursor.execute("""
                INSERT INTO classified_resumes (classified_category, similarity_score,resume_id,job_posting_id,file_path) 
                VALUES (%s, %s, %s,%s,%s)
            """, (predicted_category, similarity_score,application['r_id'],application['job_id'],application['file_path']))
        connection.commit()
        
            # Update the job_applications_data dictionary with classification results
        application['classified_category'] = predicted_category
        application['similarity_score'] = similarity_score

    cursor.close()

    return job_applications_data

def get_classified_resume_data(job_id):
    connection = mysql.connection
    cursor = connection.cursor()

    query = """
        SELECT *
        FROM classified_resumes
        where job_posting_id= %s;
    """
    
    cursor.execute(query,(job_id,))

    columns = [column[0] for column in cursor.description]
    classified_resume_data = [dict(zip(columns, row)) for row in cursor.fetchall()]

    classified_resume_data_sorted = sorted(classified_resume_data, key=lambda x: x['similarity_score'], reverse=True)

    for i, application in enumerate(classified_resume_data_sorted, start=1):
        application['rank'] = i


    cursor.close()

    return classified_resume_data_sorted


@app.route('/classified_resumes<int:job_id>', methods=['GET'])
def classified_resumes(job_id):
    if 'user_id' in session:
        classified_resume_data = get_classified_resume_data(job_id)
        return render_template('classified_resumes.html', classified_resumes=classified_resume_data)
    else:
        return redirect(url_for('login'))


@app.route('/job_applications<int:job_id>', methods=['GET'])
def job_applications(job_id):
    if 'user_id' in session:
        job_applications_data = get_job_applications_data(job_id)
        return render_template('job_applications.html', job_applications=job_applications_data,jobId= job_id)
    else :
        return redirect(url_for('login'))



##############################################################
#registerroute
# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        age = request.form['age']
        gender = request.form['gender']
        address = request.form['address']
        register_as = request.form['register_as']

        if register_as == 'candidate':
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO candidates (full_name, email, password, age, gender, address) VALUES (%s, %s, %s, %s, %s, %s)",
                (full_name, email, password, age, gender, address))
            mysql.connection.commit()
            cur.close()
        elif register_as == 'admin':
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO admins (full_name, email, password, age, gender, address) VALUES (%s, %s, %s, %s, %s, %s)",
                (full_name, email, password, age, gender, address))
            mysql.connection.commit()
            cur.close()

        return redirect(url_for('login'))  # Redirect to the login page after registration

    return render_template('register.html')

#login route
# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password_candidate = request.form['password']

        cur = mysql.connection.cursor()

        session.clear()

        # Check if the user is an admin
        result_admin = cur.execute("SELECT * FROM admins WHERE email = %s", [email])
        if result_admin > 0:
            data_admin = cur.fetchone()
            password_admin = data_admin[3]
            if bcrypt.check_password_hash(password_admin, password_candidate):
                session['user_id'] = email
                return redirect(url_for('admin_panel'))

        # Check if the user is a candidate
        result_candidate = cur.execute("SELECT * FROM candidates WHERE email = %s", [email])
        if result_candidate > 0:
            data_candidate = cur.fetchone()
            password_candidate_db = data_candidate[3]
            if bcrypt.check_password_hash(password_candidate_db, password_candidate):
                session['user_id'] = email
                return redirect(url_for('home'))

        return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')
###edit job
# ...

@app.route('/edit_job/<int:job_id>', methods=['GET', 'POST'])
def edit_job(job_id):
    if 'user_id' in session:
        user_email = session['user_id']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM admins WHERE email = %s", [user_email])

        result_admin = cur.fetchone()

        if result_admin:
            # The user is an admin, proceed with job editing

            if request.method == 'POST':
                title = request.form['title']
                description = request.form['description']
                requirements = request.form['requirements']
                salary = request.form['salary']

                cur.execute(
                    "UPDATE job_postings SET title=%s, description=%s, requirements=%s, salary=%s WHERE id=%s",
                    (title, description, requirements, salary, job_id)
                )
                mysql.connection.commit()
                cur.close()

                flash('Job posting updated successfully', 'success')
                return redirect(url_for('view_job_admin'))

            cur.execute("SELECT * FROM job_postings WHERE id = %s", [job_id])
            job_posting = cur.fetchone()
            cur.close()

            return render_template('edit_job.html', job_posting=job_posting)

        else:
            # The user is not an admin, redirect them to the home page
            return redirect(url_for('home'))

    else:
        flash('Please login to edit the job posting', 'error')
        return redirect(url_for('login'))

@app.route('/delete_job/<int:job_id>', methods=['POST'])
def delete_job(job_id):
    if 'user_id' in session:
        user_email = session['user_id']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM admins WHERE email = %s", [user_email])

        result_admin = cur.fetchone()

        if result_admin:
            # The user is an admin, proceed with job deletion

            cur.execute("DELETE FROM job_postings WHERE id = %s", [job_id])
            mysql.connection.commit()

             # Shift the IDs for remaining records
            cur.execute("SET @count = 0;")
            cur.execute("UPDATE job_postings SET id = @count:= @count + 1;")
            mysql.connection.commit()
            cur.close()

            flash('Job posting deleted successfully', 'success')
            return redirect(url_for('view_job_admin'))

        else:
            # The user is not an admin, redirect them to the home page
            return redirect(url_for('home'))

    else:
        flash('Please login to delete the job posting', 'error')
        return redirect(url_for('login'))




# ...
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.clear()
    return redirect(url_for('home'))

# main driver function
if __name__ == '__main__':
    app.run(debug=True)