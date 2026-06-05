# api/utils/validators.py

from fastapi import HTTPException


# =====================================================
# VALIDATE COURSE
# =====================================================
def validate_course(

    course,
    trained_courses

):

    if course not in trained_courses:

        raise HTTPException(

            status_code=400,

            detail={

                "error":
                    "course not trained",

                "available_courses":
                    trained_courses

            }
        )