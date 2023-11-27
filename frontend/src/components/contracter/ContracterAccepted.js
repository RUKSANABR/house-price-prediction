import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import React, { useState, useEffect } from 'react';
import { useNavigation } from '@react-navigation/native';
import firestore from '@react-native-firebase/firestore';
import { Avatar, Card } from '@rneui/base';

const ContracterAccepted = (props) => {
  const navigation = useNavigation();
  const userD = props.userD;
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const bookingDataArray = [];
        const bookingRef = firestore().collection('contract');

        const bookingQuerySnapshot = await bookingRef
          .where('contracterId', '==', userD.email)
          .where('status', '==', 'Accepted')
          .get();

        const promises = [];

        bookingQuerySnapshot.forEach((bookingDoc) => {
          const bookingData = bookingDoc.data();
          const propertyId = bookingData.propertyId;
          bookingDataArray.push({ ...bookingData, contractId: bookingDoc.id });

          const propertyRef = firestore()
            .collection('property')
            .doc(propertyId);

          // Add a promise that resolves when both the booking data and property data are fetched
          promises.push(
            Promise.all([
              propertyRef.get(),
              Promise.resolve({ ...bookingData, contractId: bookingDoc.id }),
            ]).then(([propertySnapshot, bookingData]) => {
              const propertyData = propertySnapshot.data();
              bookingData.property = propertyData;
              return bookingData;
            })
          );
        });

        // Wait for all promises to resolve before setting the state with the combined data
        Promise.all(promises)
          .then((bookingDataArray) => {
            setData(bookingDataArray);
          })
          .catch((error) => {
            console.error('Error getting documents: ', error);
          });
      } catch (error) {
        console.error('An error occurred:', error);
      }
    };

    fetchData();
  }, [userD.email]);
  return (
    <View>
      <View style={{marginBottom: 70}}>
      {data &&
          data.map((bookingData, index) =>{
            // Calculate the number of days ago
            const postedDate = new Date(bookingData.property.Date);
            const currentDate = new Date();
            const daysAgo = Math.floor((currentDate - postedDate) / (1000 * 60 * 60 * 24));
            // new code
            //
            const timestamp = new Date(bookingData.property.Date);
            const year = timestamp.getFullYear();
            const month = (timestamp.getMonth() + 1)
              .toString()
              .padStart(2, '0'); // Months are zero-based
            const day = timestamp.getDate().toString().padStart(2, '0');
            const formattedDate = `${day}-${month}-${year}`;
            // 
  
            return (
              <Card key={index} containerStyle={styles.cards}>
              <View style={styles.smallCard}>
                <View style={styles.cardContent}>
                  <Avatar source={require('../../assets/user.png')} size={30} />
                  <Text style={styles.cardText}>{bookingData.property.userName}</Text>
                </View>
                <View style={styles.cardContent}>
                  <TouchableOpacity style={styles.btn1}>
                    <Text style={styles.btnText}>{bookingData.property.LotArea} sq.ft</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.btn1}>
                    <Text style={styles.btnText}>{ formattedDate}</Text>
                  </TouchableOpacity>
                </View>

                <View style={styles.textContent}>
                      <Text style={{color: 'black'}}>{bookingData.property.TotRmsAbvGrd} Rooms</Text>
                      <View style={{flexDirection:'row',alignItems:'center'}}>
                        <Avatar source={require('../../assets/phone.png')} size={30}/>
                      <Text style={{color: 'black'}}>
                         {bookingData.property.phone}
                      </Text>
                      
                      </View>
                      <Text style={{textAlign:'justify',color:'black',marginTop:10}}>User accepted your Request.</Text>
                    </View>
                  </View>
              <View
                style={[
                  styles.cardBottomContent,
                  {justifyContent: 'space-between'},
                ]}>
                <Text style={styles.cardText}>Posted {daysAgo} days ago</Text>
                <Text style={styles.cardText}>₹{bookingData.property.price}</Text>
              </View>
            </Card>
       );
      })}
      </View>
    </View>
  );
};

export default ContracterAccepted ;

const styles = StyleSheet.create({
  cards: {
    width: '95%',
    height: 240,
    backgroundColor: 'black',
    borderTopRightRadius: 21,
    borderTopLeftRadius: 21,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    marginBottom:20,
    alignSelf:'center'
  },

  smallCard: {
    width: '108.5%',
    height: 200,
    backgroundColor: '#aedbeb',
    alignSelf: 'center',
    marginTop: -15,
    borderTopRightRadius: 20,
    borderTopLeftRadius: 20,
    padding:10
  },
  cardContent: {
    justifyContent: 'flex-start',
    flexDirection: 'row',
  },
  cardText: {
    fontSize: 18,
    fontWeight: '500',
    color: 'white',
    padding: 5,
    marginLeft: 10,
    marginRight:5
  },
  btn1: {
    backgroundColor: '#f6f0c1',
    width: 80,
    height: 39,
    borderRadius: 20,
    alignItems: 'center', // Center horizontally
    justifyContent: 'center',
    marginRight: 10,
  },
  btnText: {
    fontWeight: '500',
    color: 'white',
  },
  textContent: {
    justifyContent: 'center',
    marginLeft: 10,
    marginRight: 10,
    marginTop: 10,
  },
  cardBottomContent: {
    marginTop: 20,
    backgroundColor: 'black',
    width: '108%',
    height: 50,
    borderBottomLeftRadius: 25,
    borderBottomRightRadius: 25,
    alignSelf: 'center',
    flexDirection: 'row',
  },
});
